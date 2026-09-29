"""
Real-data archive builder: IMD observations + NOAA GFS forecasts -> MonsoonIQ schema.

Inputs (produced by the browser fetchers in scripts/fetch/, see docs/REAL_DATA.md)

    monsooniq_imd_jjas_2021-2025.bin.gz   IMD 0.25 deg daily gridded rainfall
                                           (Pai et al. 2014), 1 Jun - 30 Sep,
                                           plus GFS 0.5 deg orography + land mask
    monsooniq_gfs_jjas_<year>.bin.gz       GFS 00 UTC runs, 1 Jun - 30 Sep:
                                           Day 1-5 rainfall on the IMD 03-03 UTC
                                           day (0.5 deg) and nine dynamical
                                           predictors at 12 UTC of each lead day
                                           (1 deg)

Outputs (data/real/)

    district_daily.parquet / .csv.gz      one row per district-day, same columns
                                           as the synthetic archive, plus per-lead
                                           predictors (<field>_d<k>) and extras
    grid_feature_samples.npz              0.5 deg land-cell feature stacks for the
                                           grid correction and FSS verification
    dataset_metadata.json                 provenance, alignment test, label counts
    regime_labels.csv                     the day-level regime label per date with
                                           the index values that produced it

Design decisions (each one is written to dataset_metadata.json):

* Day definition. IMD's daily value is the 24 h accumulation ending 03 UTC. The
  GFS window is 03-03 UTC; which calendar date it maps to is measured, not
  assumed: the lag (-1/0/+1 day) that maximises the correlation between the
  all-India land-mean GFS Day-1 and IMD series is used.
* District values. Real Census-2011 district boundaries (DataMeet, CC BY 2.5 IN).
  Grid cells are weighted by the fraction of the cell inside the district
  (5 x 5 sub-sampling), so small districts such as Mumbai City are handled.
  obs_rain_mean = weighted mean; obs_rain_max = max over cells overlapping the
  district by >= 20 % (IMD's district warnings are issued on the heaviest
  rainfall in the district, not the mean).
* Regime labels. Real data has no latent regime. Labels are assigned per day from
  published objective criteria (Rajeevan et al. 2010 active/break index on IMD
  core-monsoon-zone rainfall; depression/low from GFS MSLP + 850 hPa vorticity;
  offshore-trough orographic spells; east-coast spells; western disturbances),
  see configs/regime_labels_real.yaml. The classifier then has to recover these
  labels from GFS dynamics alone - a real test, unlike the synthetic archive.
"""

import gzip
import io
import json
import os
import struct
import logging
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.compat import points_in_polygon, write_table

logger = logging.getLogger("MonsoonIQ.real_archive")

IMD_LAT0, IMD_LON0, IMD_RES, IMD_NLAT, IMD_NLON = 6.5, 66.5, 0.25, 129, 135
G05_LATS = np.arange(8.0, 36.0 + 1e-9, 0.5)          # 57, ascending (pipeline order)
G05_LONS = np.arange(68.0, 96.0 + 1e-9, 0.5)          # 57
EARTH_R = 6.371e6
R_ = np.pi / 180.0


# ------------------------------------------------------------------ readers
def _read_bundle(path: str, magic: bytes) -> Tuple[dict, bytes]:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != magic:
        raise ValueError(f"{path}: bad magic {raw[:4]!r}")
    hlen = struct.unpack(">I", raw[4:8])[0]
    header = json.loads(raw[8:8 + hlen].decode("utf-8"))
    return header, raw[8 + hlen:]


def _decide_endian(block: bytes) -> str:
    best, best_score = "<f4", -1.0
    for dt in ("<f4", ">f4"):
        a = np.frombuffer(block[: 4 * IMD_NLAT * IMD_NLON * 10], dtype=dt)
        ok = np.isfinite(a) & ((a == -999.0) | ((a >= 0) & (a < 1500)))
        land = a[ok & (a != -999.0)]
        score = ok.mean() + (0.5 if land.size and 0.05 < land.mean() < 60 else 0.0)
        if score > best_score:
            best, best_score = dt, score
    return best


def load_imd(path: str) -> Tuple[Dict[int, np.ndarray], List[date], dict]:
    """IMD bundle -> ({year: (122, 129, 135) mm, NaN outside India}, dates, static)."""
    header, body = _read_bundle(path, b"MIQI")
    per = 122 * IMD_NLAT * IMD_NLON * 4
    out, dates = {}, []
    endian = None
    for k, y in enumerate(header["years"]):
        block = body[k * per:(k + 1) * per]
        endian = endian or _decide_endian(block)
        a = np.frombuffer(block, dtype=endian).astype(np.float32).reshape(122, IMD_NLAT, IMD_NLON)
        a = np.where((a < -1) | ~np.isfinite(a), np.nan, a)
        out[int(y)] = a
        dates += [date(int(y), 6, 1) + timedelta(days=i) for i in range(122)]
    header["endian_detected"] = endian
    return out, dates, header


def load_gfs(path: str) -> dict:
    header, body = _read_bundle(path, b"MIQ1")
    n = len(header["dates"])
    ps = header["precip"]["shape"]
    qs = header["pred"]["shape"]
    np_ = int(np.prod(ps)) * 4
    precip = np.frombuffer(body[:np_], dtype="<f4").reshape(ps)
    pred = np.frombuffer(body[np_:np_ + int(np.prod(qs)) * 4], dtype="<f4").reshape(qs)
    # stored north->south; flip precip to the pipeline's ascending latitude order
    precip = precip[:, :, ::-1, :]
    pred = pred[:, :, :, ::-1, :]
    return {"header": header, "dates": [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in header["dates"]],
            "precip": precip, "pred": pred,
            "pred_vars": [v.split("@")[0] for v in header["pred"]["vars"]],
            "pred_lats": np.arange(0.0, 40.0 + 1e-9, 1.0), "pred_lons": np.arange(50.0, 100.0 + 1e-9, 1.0),
            "n": n}


# --------------------------------------------------------------- geometry
def cell_weights(rings, lats: np.ndarray, lons: np.ndarray, res: float, sub: int = 5) -> np.ndarray:
    """Fraction of each grid cell (centres lats x lons) inside a polygon."""
    allpts = np.vstack(rings)
    x0, x1 = allpts[:, 0].min() - res, allpts[:, 0].max() + res
    y0, y1 = allpts[:, 1].min() - res, allpts[:, 1].max() + res
    ii = np.where((lats >= y0) & (lats <= y1))[0]
    jj = np.where((lons >= x0) & (lons <= x1))[0]
    w = np.zeros((len(lats), len(lons)), dtype=np.float32)
    if len(ii) == 0 or len(jj) == 0:
        return w
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    la, lo = np.meshgrid(lats[ii], lons[jj], indexing="ij")
    tot = np.zeros(la.shape, dtype=np.float32)
    for dy in offs:
        for dx in offs:
            tot += points_in_polygon(rings, lo + dx * res, la + dy * res)
    w[np.ix_(ii, jj)] = tot / (sub * sub)
    return w


def bilinear(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, qlat, qlon):
    """Bilinear interpolation on a regular ascending grid (last two axes)."""
    qlat, qlon = np.asarray(qlat, float), np.asarray(qlon, float)
    fi = (qlat - lats[0]) / (lats[1] - lats[0])
    fj = (qlon - lons[0]) / (lons[1] - lons[0])
    i0 = np.clip(np.floor(fi).astype(int), 0, len(lats) - 2)
    j0 = np.clip(np.floor(fj).astype(int), 0, len(lons) - 2)
    ti, tj = fi - i0, fj - j0
    f = field
    return ((1 - ti) * (1 - tj) * f[..., i0, j0] + ti * (1 - tj) * f[..., i0 + 1, j0]
            + (1 - ti) * tj * f[..., i0, j0 + 1] + ti * tj * f[..., i0 + 1, j0 + 1])


def regrid_to(field: np.ndarray, lats, lons, new_lats, new_lons) -> np.ndarray:
    la, lo = np.meshgrid(new_lats, new_lons, indexing="ij")
    return bilinear(field, lats, lons, la, lo)


def vorticity(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Relative vorticity dv/dx - du/dy (s^-1), centred differences, last two axes lat, lon."""
    dlat = (lats[1] - lats[0]) * R_ * EARTH_R
    dlon = (lons[1] - lons[0]) * R_ * EARTH_R * np.cos(lats * R_)[:, None]
    dvdx = np.gradient(v, axis=-1) / dlon
    dudy = np.gradient(u * np.cos(lats * R_)[:, None], axis=-2) / dlat / np.cos(lats * R_)[:, None]
    return dvdx - dudy


def imd_to_05(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Area-consistent 0.25 -> 0.5 deg (1-2-1 tent over the co-located centres).

    Returns (field on G05 grid, land fraction on G05 grid)."""
    imd_lats = IMD_LAT0 + IMD_RES * np.arange(IMD_NLAT)
    imd_lons = IMD_LON0 + IMD_RES * np.arange(IMD_NLON)
    ii = np.searchsorted(imd_lats, G05_LATS - 1e-6)
    jj = np.searchsorted(imd_lons, G05_LONS - 1e-6)
    wts = np.array([0.25, 0.5, 0.25])
    lead = a.shape[:-2]
    num = np.zeros(lead + (len(G05_LATS), len(G05_LONS)), dtype=np.float64)
    den = np.zeros_like(num)
    valid = np.isfinite(a)
    a0 = np.where(valid, a, 0.0)
    for di, wi in zip((-1, 0, 1), wts):
        for dj, wj in zip((-1, 0, 1), wts):
            I = np.clip(ii + di, 0, IMD_NLAT - 1)
            J = np.clip(jj + dj, 0, IMD_NLON - 1)
            num += wi * wj * a0[..., I[:, None], J[None, :]]
            den += wi * wj * valid[..., I[:, None], J[None, :]]
    out = np.where(den >= 0.5, num / np.maximum(den, 1e-9), np.nan)
    return out.astype(np.float32), den


# ----------------------------------------------------------------- build
class RealArchiveBuilder:
    LEADS = (1, 2, 3, 4, 5)

    def __init__(self, bundle_dir: str, out_dir: str = "data/real",
                 geojson_path: str = "data/geojson/india_districts.geojson",
                 boundaries_path: str = "data/geojson/india_districts_census2011.geojson",
                 labels_cfg: str = "configs/regime_labels_real.yaml"):
        self.bundle_dir = bundle_dir
        self.out_dir = out_dir
        with open(geojson_path, encoding="utf-8") as f:
            self.meta_geo = json.load(f)
        with open(boundaries_path, encoding="utf-8") as f:
            self.boundaries = {ft["properties"]["district_id"]: ft for ft in json.load(f)["features"]}
        with open(labels_cfg, encoding="utf-8") as f:
            self.label_cfg = yaml.safe_load(f)
        self.meta: dict = {}

    # -- inputs
    def _load(self):
        imd_path = os.path.join(self.bundle_dir, "monsooniq_imd_jjas_2021-2025.bin.gz")
        self.imd, self.imd_dates, self.imd_header = load_imd(imd_path)
        self.years = sorted(self.imd)
        self.gfs = {}
        for y in self.years:
            p = os.path.join(self.bundle_dir, f"monsooniq_gfs_jjas_{y}.bin.gz")
            if os.path.exists(p):
                self.gfs[y] = load_gfs(p)
            else:
                logger.warning("missing GFS bundle for %d", y)
        self.years = [y for y in self.years if y in self.gfs]
        st = self.imd_header["static"]
        n = st["grid"]["n"]
        slats = st["grid"]["lat_first"] + st["grid"]["lat_step"] * np.arange(n)
        slons = st["grid"]["lon_first"] + st["grid"]["lon_step"] * np.arange(n)
        self.st_lats, self.st_lons = slats[::-1], slons
        self.hgt = np.asarray(st["hgt"], np.float32).reshape(n, n)[::-1]
        self.land = np.asarray(st["land"], np.float32).reshape(n, n)[::-1]

    # -- static terrain
    def _terrain(self):
        dy = 0.5 * R_ * EARTH_R
        dx = 0.5 * R_ * EARTH_R * np.cos(self.st_lats * R_)[:, None]
        gy, gx = np.gradient(self.hgt)
        self.slope_grid = np.sqrt((gx / dx) ** 2 + (gy / dy) ** 2)
        la, lo = np.meshgrid(self.st_lats, self.st_lons, indexing="ij")
        sea = self.land < 0.5
        sla, slo = la[sea], lo[sea]

        def dist_coast(qlat, qlon):
            d = EARTH_R / 1000 * np.arccos(np.clip(
                np.sin(qlat * R_) * np.sin(sla * R_) + np.cos(qlat * R_) * np.cos(sla * R_) * np.cos((qlon - slo) * R_), -1, 1))
            return float(max(0.0, d.min() - 27.0))  # distance to the sea-cell edge
        self.dist_coast = dist_coast

    # -- day alignment between IMD dates and GFS 03-03 UTC windows
    def _alignment(self):
        imd_lats = IMD_LAT0 + IMD_RES * np.arange(IMD_NLAT)
        imd_lons = IMD_LON0 + IMD_RES * np.arange(IMD_NLON)
        scores = {}
        for lag in (-1, 0, 1):
            xs, ys = [], []
            for y in self.years:
                g = self.gfs[y]
                obs05, _ = imd_to_05(self.imd[y])
                land = np.isfinite(obs05[0])
                for k, d in enumerate(g["dates"]):
                    t = (d - date(y, 6, 1)).days + lag
                    if 0 <= t < 122:
                        xs.append(np.nanmean(g["precip"][k, 0][land]))
                        ys.append(np.nanmean(obs05[t][land]))
            xs, ys = np.asarray(xs), np.asarray(ys)
            ok = np.isfinite(xs) & np.isfinite(ys)
            scores[lag] = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        self.lag = max(scores, key=scores.get)
        self.meta["day_alignment"] = {
            "definition": "GFS window 03-03 UTC starting on run date D; IMD value for date D+lag",
            "correlation_by_lag": scores, "chosen_lag_days": self.lag,
            "note": "all-India land-mean Day-1 GFS vs IMD, all seasons"}
        logger.info("Day alignment correlations %s -> lag %d", scores, self.lag)

    # -- district weights
    def _district_weights(self):
        imd_lats = IMD_LAT0 + IMD_RES * np.arange(IMD_NLAT)
        imd_lons = IMD_LON0 + IMD_RES * np.arange(IMD_NLON)
        self.districts = []
        for ft in self.meta_geo["features"]:
            p = ft["properties"]
            b = self.boundaries[p["district_id"]]
            geom = b["geometry"]
            polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
            rings = [np.asarray(r) for poly in polys for r in poly]
            w_imd = cell_weights(rings, imd_lats, imd_lons, IMD_RES)
            land = np.isfinite(self.imd[self.years[0]][0])
            if not (land & (w_imd > 0)).any():
                # Coastal city districts (e.g. Chennai) can overlap only sea cells in
                # the IMD land-only grid: use the 4 nearest land cells instead.
                la, lo = np.meshgrid(imd_lats, imd_lons, indexing="ij")
                d2 = (la - p["centroid_lat"]) ** 2 + (lo - p["centroid_lon"]) ** 2
                d2 = np.where(land, d2, np.inf)
                near = np.argsort(d2.ravel())[:4]
                w_imd = np.zeros_like(w_imd)
                w_imd.ravel()[near] = 1.0
                self.meta.setdefault("nearest_land_cell_fallback", []).append(p["district_id"])
            w_gfs = cell_weights(rings, G05_LATS, G05_LONS, 0.5)
            if w_gfs.sum() == 0:  # guard for a district smaller than any sub-sample
                i = np.argmin(abs(G05_LATS - p["centroid_lat"]))
                j = np.argmin(abs(G05_LONS - p["centroid_lon"]))
                w_gfs[i, j] = 1.0
            self.districts.append({"p": p, "w_imd": w_imd, "w_gfs": w_gfs,
                                   "area_cells_imd": float(w_imd.sum())})

    # -- day-level indices from one set of predictor fields
    def _day_indices(self, fld: Dict[str, np.ndarray], lats, lons) -> Dict[str, float]:
        def box(a, la0, la1, lo0, lo1):
            i = (lats >= la0) & (lats <= la1)
            j = (lons >= lo0) & (lons <= lo1)
            return a[np.ix_(i, j)]
        mslp = fld["PRMSL"] / 100.0
        zon = box(mslp, 15, 32, 75, 85).mean(axis=1)
        tlat = lats[(lats >= 15) & (lats <= 32)][int(np.argmin(zon))]
        vort = vorticity(fld["UGRD"], fld["VGRD"], lats, lons)
        # 3x3 smoothing before taking the vortex maximum
        from scipy.ndimage import uniform_filter
        vs = uniform_filter(vort, size=3, mode="nearest")
        return {
            "olr_cmz": float(box(fld["ULWRF"], 18, 26, 74, 86).mean()),
            "mslp_min_box": float(box(mslp, 15, 26, 78, 92).min()),
            "vort_max_box": float(box(vs, 15, 26, 78, 92).max()),
            "trough_latitude": float(tlat),
            "u850_westcoast": float(box(fld["UGRD"], 12, 20, 70, 74).mean()),
            "z500_nw": float(box(fld["HGT"], 30, 37, 68, 78).mean()),
        }

    def build(self) -> pd.DataFrame:
        self._load()
        self._terrain()
        self._alignment()
        self._district_weights()
        cfg = self.label_cfg

        # ---- observations on district + 0.5 grid for every IMD date
        obs_rows = {}
        obs05_all = {}
        for y in self.years:
            obs05, _ = imd_to_05(self.imd[y])
            for t in range(122):
                d = date(y, 6, 1) + timedelta(days=t)
                obs05_all[d] = obs05[t]
                a = self.imd[y][t]
                rec = {}
                for dd in self.districts:
                    w = dd["w_imd"]
                    m = np.isfinite(a) & (w > 0)
                    if not m.any():
                        rec[dd["p"]["district_id"]] = (np.nan, np.nan)
                        continue
                    mean = float((a[m] * w[m]).sum() / w[m].sum())
                    big = m & (w >= 0.2)
                    mx = float(np.nanmax(a[big])) if big.any() else float(np.nanmax(a[m]))
                    rec[dd["p"]["district_id"]] = (mean, mx)
                obs_rows[d] = rec

        # ---- GFS per valid date and lead
        run_index = {}
        for y in self.years:
            g = self.gfs[y]
            for k, d in enumerate(g["dates"]):
                run_index[d] = (y, k)

        def gfs_for(valid: date, lead: int):
            """Forecast valid on IMD date `valid` at lead `lead` (run = valid-lag-(lead-1))."""
            run = valid - timedelta(days=self.lag + lead - 1)
            if run not in run_index:
                return None
            y, k = run_index[run]
            g = self.gfs[y]
            precip = g["precip"][k, lead - 1]
            if not np.isfinite(precip).all():
                return None
            fld = {v: g["pred"][k, lead - 1, i] for i, v in enumerate(g["pred_vars"])}
            if not all(np.isfinite(a).all() for a in fld.values()):
                return None
            return precip, fld, g["pred_lats"], g["pred_lons"]

        valid_dates = sorted(d for d in obs_rows if all(gfs_for(d, L) is not None for L in self.LEADS))
        logger.info("%d valid dates with observations and all 5 leads", len(valid_dates))

        # ---- climatologies for anomalies (training years only, smoothed seasonal cycle)
        from src.config import SPLITS
        train_years = set(SPLITS["real"]["train"])
        idx_rows = []
        for d in valid_dates:
            _, fld, la, lo = gfs_for(d, 1)
            ix = self._day_indices(fld, la, lo)
            ix["date"] = d
            idx_rows.append(ix)
        idx = pd.DataFrame(idx_rows).set_index("date")
        doy = np.array([(d - date(d.year, 6, 1)).days for d in idx.index])

        def seasonal_clim(series: pd.Series) -> np.ndarray:
            s = series.values
            mask = np.array([d.year in train_years for d in series.index])
            clim = np.full(122, np.nan)
            for t in range(122):
                sel = mask & (np.abs(doy - t) <= 15)
                clim[t] = np.nanmean(s[sel]) if sel.any() else np.nanmean(s[mask])
            return clim

        clim = {c: seasonal_clim(idx[c]) for c in ("olr_cmz", "mslp_min_box", "z500_nw")}
        self.meta["climatology"] = {"years": sorted(train_years), "window_days": 31,
                                    "note": "anomalies use a training-years-only seasonal cycle"}

        # ---- regime labels (day level)
        labels = self._labels(valid_dates, idx, clim, doy, obs_rows)

        # ---- district rows
        rows = []
        for n_d, d in enumerate(valid_dates):
            t = (d - date(d.year, 6, 1)).days
            per_lead = {L: gfs_for(d, L) for L in self.LEADS}
            lead_fields = {}
            for L, (precip, fld, la, lo) in per_lead.items():
                u, v = fld["UGRD"], fld["VGRD"]
                lead_fields[L] = {
                    "precip": precip, "u850": u, "v850": v,
                    "vorticity_850": vorticity(u, v, la, lo), "q500": fld["SPFH"],
                    "cape": fld["CAPE"], "olr": fld["ULWRF"], "rh850": fld["RH"],
                    "pwat": fld["PWAT"], "mslp": fld["PRMSL"] / 100.0, "z500": fld["HGT"],
                    "la": la, "lo": lo,
                    "idx": self._day_indices(fld, la, lo),
                }
            lab = labels.loc[d]
            for dd in self.districts:
                p = dd["p"]
                did = p["district_id"]
                om, ox = obs_rows[d][did]
                if not np.isfinite(om):
                    continue
                clat, clon = p["centroid_lat"], p["centroid_lon"]
                row = {
                    "date": d.isoformat(), "year": d.year, "month": d.month, "doy_season": t,
                    "district_id": did, "district_name": p["district_name"],
                    "state_name": p["state_name"], "zone": p["zone"],
                    "regime": int(lab["regime"]), "regime_name": lab["regime_name"],
                    "obs_rain_mean": om, "obs_rain_max": ox,
                    "obs_heavy": int(ox >= 64.5), "obs_very_heavy": int(ox >= 115.6),
                    "obs_extremely_heavy": int(ox >= 204.5),
                    "elevation": float(p.get("elevation_m", 0.0)),
                    "slope": float(bilinear(self.slope_grid, self.st_lats, self.st_lons, clat, clon)),
                    "dist_coast": self.dist_coast(clat, clon),
                    "latitude": clat, "longitude": clon,
                }
                for L, lf in lead_fields.items():
                    w = dd["w_gfs"]
                    row[f"raw_nwp_d{L}"] = float((lf["precip"] * w).sum() / w.sum())
                    vals = {f: float(bilinear(lf[f], lf["la"], lf["lo"], clat, clon))
                            for f in ("u850", "v850", "vorticity_850", "q500", "cape", "olr",
                                      "rh850", "pwat", "mslp", "z500")}
                    vals["wind_speed_850"] = float(np.hypot(vals["u850"], vals["v850"]))
                    vals["moisture_flux"] = vals["wind_speed_850"] * vals["q500"] * 1000.0
                    vals["olr_anomaly"] = lf["idx"]["olr_cmz"] - clim["olr_cmz"][t]
                    vals["mslp_anomaly"] = lf["idx"]["mslp_min_box"] - clim["mslp_min_box"][t]
                    vals["trough_latitude"] = lf["idx"]["trough_latitude"]
                    # day-level synoptic indices from the same forecast (domain scale)
                    vals["vort_max_box"] = lf["idx"]["vort_max_box"]
                    vals["u850_westcoast"] = lf["idx"]["u850_westcoast"]
                    vals["z500_anomaly_nw"] = lf["idx"]["z500_nw"] - clim["z500_nw"][t]
                    for f, v in vals.items():
                        row[f"{f}_d{L}"] = v
                        if L == 1:
                            row[f] = v  # the row's own predictors = Day-1 forecast of its valid day
                rows.append(row)
            if n_d % 100 == 0:
                logger.info("  district rows: %d / %d dates", n_d, len(valid_dates))

        df = pd.DataFrame(rows)
        self._write(df, valid_dates, per_lead_cache=None, obs05_all=obs05_all,
                    gfs_for=gfs_for, labels=labels, clim=clim, idx=idx)
        return df

    # ------------------------------------------------------------------ labels
    def _labels(self, valid_dates, idx, clim, doy, obs_rows) -> pd.DataFrame:
        cfg = self.label_cfg
        names = {int(k): v["name"] for k, v in cfg["regimes"].items()}
        train_years = set(cfg.get("climatology_years", []))

        def zone_mean(d, zones):
            vals = [obs_rows[d][dd["p"]["district_id"]][0] for dd in self.districts
                    if dd["p"]["zone"] in zones and np.isfinite(obs_rows[d][dd["p"]["district_id"]][0])]
            return float(np.mean(vals)) if vals else np.nan

        # core monsoon zone rainfall (Rajeevan et al. 2010) from the IMD grid
        imd_lats = IMD_LAT0 + IMD_RES * np.arange(IMD_NLAT)
        imd_lons = IMD_LON0 + IMD_RES * np.arange(IMD_NLON)
        c = cfg["active_break"]
        mi = (imd_lats >= c["box"][0]) & (imd_lats <= c["box"][1])
        mj = (imd_lons >= c["box"][2]) & (imd_lons <= c["box"][3])
        cmz = []
        for d in valid_dates:
            t = (d - date(d.year, 6, 1)).days
            a = self.imd[d.year][t][np.ix_(mi, mj)]
            cmz.append(float(np.nanmean(a)))
        s = pd.DataFrame({"cmz": cmz,
                          "west": [zone_mean(d, ["West Coast"]) for d in valid_dates],
                          "east": [zone_mean(d, ["East Coast"]) for d in valid_dates]},
                         index=valid_dates)

        from src.config import SPLITS
        fit_years = set(SPLITS["real"]["train"])

        def standardise(col):
            # seasonal cycle and spread from the training seasons only, so no
            # held-out season shapes the labels it is scored against
            v = s[col].values
            yrs = np.array([d.year for d in s.index])
            fit = np.isin(yrs, list(fit_years))
            clim_ = np.full(122, np.nan)
            for t in range(122):
                sel = fit & (np.abs(doy - t) <= 15)
                clim_[t] = np.nanmean(v[sel])
            anom = v - clim_[doy]
            return anom / np.nanstd(anom[fit])

        for col in ("cmz", "west", "east"):
            s[col + "_z"] = standardise(col)

        def spells(mask: np.ndarray, min_len: int) -> np.ndarray:
            out = np.zeros_like(mask)
            run = 0
            dates = list(s.index)
            for i in range(len(mask)):
                contiguous = i > 0 and (dates[i] - dates[i - 1]).days == 1
                run = run + 1 if (mask[i] and (contiguous or run == 0)) else (1 if mask[i] else 0)
                if run >= min_len:
                    out[i - run + 1:i + 1] = True
            return out

        months = np.array([d.month for d in s.index])
        ab_months = np.isin(months, c["months"])
        active = spells((s["cmz_z"].values >= c["threshold_sd"]) & ab_months, c["min_days"])
        brk = spells((s["cmz_z"].values <= -c["threshold_sd"]) & ab_months, c["min_days"])

        dep_c = cfg["depression"]
        mslp_anom = idx["mslp_min_box"].values - clim["mslp_min_box"][doy]
        dep = (mslp_anom <= dep_c["mslp_anomaly_max_hpa"]) & (idx["vort_max_box"].values >= dep_c["vorticity_min"])

        oro_c = cfg["orographic"]
        oro = (idx["u850_westcoast"].values >= oro_c["u850_min"]) & (s["west_z"].values >= oro_c["west_coast_rain_z_min"])
        cst_c = cfg["coastal"]
        cst = s["east_z"].values >= cst_c["east_coast_rain_z_min"]
        wd_c = cfg["western_disturbance"]
        wd = (idx["z500_nw"].values - clim["z500_nw"][doy]) <= wd_c["z500_anomaly_max_gpm"]

        regime = np.full(len(s), 7)
        # lowest priority first; later assignments override
        for mask, code in ((wd, 6), (cst, 5), (oro, 4), (active, 1), (brk, 2), (dep, 3)):
            regime = np.where(mask, code, regime)
        out = pd.DataFrame({"regime": regime, "regime_name": [names[r] for r in regime],
                            "cmz_rain_z": s["cmz_z"].values, "west_coast_rain_z": s["west_z"].values,
                            "east_coast_rain_z": s["east_z"].values, "mslp_anomaly_box_hpa": mslp_anom,
                            "vort_max_box": idx["vort_max_box"].values,
                            "u850_westcoast": idx["u850_westcoast"].values,
                            "z500_nw_anomaly_gpm": idx["z500_nw"].values - clim["z500_nw"][doy],
                            "trough_latitude": idx["trough_latitude"].values}, index=s.index)
        counts = out["regime_name"].value_counts().to_dict()
        self.meta["regime_labels"] = {"source": "configs/regime_labels_real.yaml", "day_counts": counts}
        logger.info("Regime label counts: %s", counts)
        return out

    # ------------------------------------------------------------------ write
    def _write(self, df, valid_dates, per_lead_cache, obs05_all, gfs_for, labels, clim, idx):
        os.makedirs(self.out_dir, exist_ok=True)
        archive = os.path.join(self.out_dir, "district_daily.parquet")
        written = write_table(df, archive)
        labels.reset_index().rename(columns={"index": "date"}).to_csv(
            os.path.join(self.out_dir, "regime_labels.csv"), index=False, float_format="%.5g")

        # grid archive: every held-out day, every 3rd day otherwise
        from src.config import SPLITS
        test_years = set(SPLITS["real"]["test"])
        land05 = np.isfinite(obs05_all[valid_dates[0]])
        for d in valid_dates[:20]:
            land05 &= np.isfinite(obs05_all[d])
        land_index = np.flatnonzero(land05.ravel()).astype(np.int32)
        la2, lo2 = np.meshgrid(G05_LATS, G05_LONS, indexing="ij")
        elev05 = regrid_to(self.hgt, self.st_lats, self.st_lons, G05_LATS, G05_LONS)
        slope05 = regrid_to(self.slope_grid, self.st_lats, self.st_lons, G05_LATS, G05_LONS)
        dc = np.array([self.dist_coast(a, b) for a, b in zip(la2.ravel()[land_index], lo2.ravel()[land_index])], np.float32)
        records = {}
        for n, d in enumerate(valid_dates):
            if d.year not in test_years and n % 3:
                continue
            t = (d - date(d.year, 6, 1)).days
            p1, f1, la, lo = gfs_for(d, 1)
            p3 = gfs_for(d, 3)[0]
            u, v = f1["UGRD"], f1["VGRD"]
            fields = {"u850": u, "v850": v, "wind_speed_850": np.hypot(u, v),
                      "vorticity_850": vorticity(u, v, la, lo), "q500": f1["SPFH"],
                      "cape": f1["CAPE"], "olr": f1["ULWRF"]}
            rec = {k: regrid_to(a, la, lo, G05_LATS, G05_LONS).ravel()[land_index].astype(np.float32)
                   for k, a in fields.items()}
            rec["moisture_flux"] = (rec["wind_speed_850"] * rec["q500"] * 1000.0).astype(np.float32)
            rec["true_rain"] = obs05_all[d].ravel()[land_index].astype(np.float32)
            rec["raw_nwp_d1"] = p1.ravel()[land_index].astype(np.float32)
            rec["raw_nwp_d3"] = p3.ravel()[land_index].astype(np.float32)
            ix = idx.loc[d]
            for k, val in (("mslp_anomaly", ix["mslp_min_box"] - clim["mslp_min_box"][t]),
                           ("trough_latitude", ix["trough_latitude"]),
                           ("olr_anomaly", ix["olr_cmz"] - clim["olr_cmz"][t])):
                rec[k] = np.full(len(land_index), val, np.float32)
            rec["elevation"] = elev05.ravel()[land_index].astype(np.float32)
            rec["slope"] = slope05.ravel()[land_index].astype(np.float32)
            rec["dist_coast"] = dc
            rec["latitude"] = la2.ravel()[land_index].astype(np.float32)
            rec["longitude"] = lo2.ravel()[land_index].astype(np.float32)
            rec["regime_scalar"] = np.int16(labels.loc[d, "regime"])
            records[d.isoformat()] = rec
        np.savez_compressed(os.path.join(self.out_dir, "grid_feature_samples.npz"),
                            lats=G05_LATS.astype(np.float32), lons=G05_LONS.astype(np.float32),
                            land_index=land_index, grid_shape=np.array([57, 57], np.int32),
                            elevation_grid=elev05.astype(np.float32), land_mask=land05.astype(np.float32),
                            **{k: np.array(v, dtype=object) for k, v in records.items()})

        self.meta.update({
            "dataset_type": "REAL_IMD_GFS",
            "provenance_note": ("Observations: IMD 0.25 deg daily gridded rainfall (Pai et al. 2014, IMD Pune). "
                                "Forecasts: NOAA GFS 00 UTC runs (AWS Open Data noaa-gfs-bdp-pds), Day 1-5. "
                                "District boundaries: DataMeet Census-2011 (CC BY 2.5 IN). "
                                "GFS stands in for NCMRWF NCUM, which is not publicly downloadable."),
            "years": self.years, "season": "1 Jun - 30 Sep",
            "total_days": len(valid_dates), "total_district_records": int(len(df)),
            "districts_count": len(self.districts),
            "grid_dates": len(records),
            "domain": {"lat_min": 8.0, "lat_max": 36.0, "lon_min": 68.0, "lon_max": 96.0,
                       "resolution_deg": 0.5},
            "imd_endian": self.imd_header.get("endian_detected"),
            "gfs_missing_fields": {y: len(self.gfs[y]["header"].get("missing", [])) for y in self.years},
            "archive_file": os.path.basename(written),
            "regimes_modeled": [v["name"] for _, v in sorted(self.label_cfg["regimes"].items())],
        })
        with open(os.path.join(self.out_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, default=str)
        logger.info("Wrote %d rows, %d grid dates to %s", len(df), len(records), self.out_dir)
