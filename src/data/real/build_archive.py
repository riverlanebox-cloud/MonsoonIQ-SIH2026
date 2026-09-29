"""
Assemble the real district archive - same schema as the synthetic one, so the entire
train -> evaluate -> console -> API chain runs unchanged with MONSOONIQ_PROFILE=real.

Stages (each writes a CSV under data/real/interim/, so any stage can be re-run alone):
  observations  IMD 0.25 deg grid -> district mean / max / p90 (area-weighted on Census-2011
                polygons) + core-monsoon-zone rainfall series for the active/break index
  forecast      raw NWP Day 1..5 per district (Open-Meteo Previous Runs, or gridded NetCDF)
  dynamics      issue-time (00 UTC) predictors + IMD-day means for labels (Open-Meteo or ERA5)
  olr           NOAA interpolated OLR + anomaly
  terrain       DEM elevation and slope at the district representative point
  assemble      join, derive anomalies / trough latitude / moisture flux, label regimes,
                write data/real/district_daily.(parquet|csv.gz) + dataset_metadata.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data import geometry as G
from src.data.io import write_table
from src.data.real import imd
from src.regime import real_labeller

logger = logging.getLogger(__name__)

STUDY = "data/geojson/india_districts.geojson"
ALL = "data/geojson/india_districts_census2011.geojson"

ZONE_BY_STATE = {
    "Kerala": "West Coast", "Karnataka": "South Peninsula", "Goa": "West Coast", "Maharashtra": "Central India",
    "Gujarat": "Northwest", "Rajasthan": "Northwest", "Punjab": "Northwest", "Haryana": "Northwest",
    "NCT of Delhi": "Northwest", "Delhi": "Northwest", "Himachal Pradesh": "Northwest",
    "Jammu & Kashmir": "Northwest", "Jammu and Kashmir": "Northwest", "Uttarakhand": "Northwest",
    "Uttar Pradesh": "Gangetic Plain", "Bihar": "Gangetic Plain", "Jharkhand": "Central India",
    "Madhya Pradesh": "Central India", "Chhattisgarh": "Central India", "Odisha": "East Coast",
    "West Bengal": "East Coast", "Andhra Pradesh": "East Coast", "Telangana": "South Peninsula",
    "Tamil Nadu": "South Peninsula", "Puducherry": "South Peninsula", "Assam": "Northeast",
    "Meghalaya": "Northeast", "Arunachal Pradesh": "Northeast", "Nagaland": "Northeast",
    "Manipur": "Northeast", "Mizoram": "Northeast", "Tripura": "Northeast", "Sikkim": "Northeast",
}


# ------------------------------------------------------------------ districts
def load_districts(scope: str = "study") -> List[dict]:
    path = STUDY if scope == "study" else ALL
    with open(path, "r", encoding="utf-8") as f:
        feats = json.load(f)["features"]
    for ft in feats:
        p = ft["properties"]
        if "rep_lat" not in p:
            lon, lat = G.representative_point(ft["geometry"])
            p["rep_lat"], p["rep_lon"] = lat, lon
        p.setdefault("zone", ZONE_BY_STATE.get(p.get("state_name", ""), "Central India"))
    return feats


def sample_points(feats: List[dict], per_district: int = 1) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Representative point, plus (per_district-1) interior points spread over the polygon."""
    pts, owner = [], []
    for ft in feats:
        p = ft["properties"]
        pts.append((p["rep_lat"], p["rep_lon"]))
        owner.append(p["district_id"])
        if per_district > 1:
            x0, y0, x1, y1 = G.bounds(ft["geometry"])
            gx, gy = np.meshgrid(np.linspace(x0, x1, 12), np.linspace(y0, y1, 12))
            m = G.contains(ft["geometry"], gx.ravel(), gy.ravel())
            cand = np.column_stack([gy.ravel()[m], gx.ravel()[m]])
            if len(cand):
                pick = cand[np.linspace(0, len(cand) - 1, per_district - 1).astype(int)]
                for lat, lon in pick:
                    pts.append((float(lat), float(lon)))
                    owner.append(p["district_id"])
    return pts, owner


# ------------------------------------------------------------------ stages
def stage_observations(raw_dir: str, feats: List[dict], days: List[date], out_dir: str) -> pd.DataFrame:
    fields, found = imd.load_days(os.path.join(raw_dir, "imd"), days)
    if not found:
        raise FileNotFoundError("no IMD rainfall found - run `fetch_real_data.py imd` first")
    weights = {ft["properties"]["district_id"]: G.cell_weights(ft["geometry"], imd.LATS, imd.LONS, imd.RES)
               for ft in feats}
    # Small coastal districts (e.g. Chennai, Mumbai City) can overlap only cells IMD leaves
    # missing over the sea; fall back to the nearest land cells that carry data.
    land = ~np.isnan(np.nanmax(fields, axis=0)) if len(found) else None
    for ft in feats:
        did = ft["properties"]["district_id"]
        idx, w = weights[did]
        if land is not None and not any(land[i, j] for i, j in idx):
            lon, lat = G.representative_point(ft["geometry"])
            li, lj = np.where(land)
            d2 = (imd.LATS[li] - lat) ** 2 + ((imd.LONS[lj] - lon) * np.cos(np.radians(lat))) ** 2
            near = np.argsort(d2)[:2]
            weights[did] = ([(int(li[k]), int(lj[k])) for k in near], np.array([0.5, 0.5], dtype=np.float32))
            logger.info("%s: no IMD land cell inside polygon - using the 2 nearest land cells", did)
    rows = []
    for k, d in enumerate(found):
        f = fields[k]
        for did, (idx, w) in weights.items():
            vals = np.array([f[i, j] for i, j in idx], dtype=float)
            ok = ~np.isnan(vals)
            if not ok.any():
                continue
            ww = w[ok] / w[ok].sum()
            v = vals[ok]
            major = v[w[ok] >= 0.25] if (w[ok] >= 0.25).any() else v   # ignore sliver cells for max
            rows.append({"date": d.isoformat(), "district_id": did,
                         "obs_rain_mean": round(float(np.sum(v * ww)), 2),
                         "obs_rain_max": round(max(float(np.max(major)), float(np.sum(v * ww))), 2),
                         "obs_rain_p90": round(float(np.percentile(v, 90)), 2),
                         "obs_cells": int(ok.sum())})
    obs = pd.DataFrame(rows)
    cmz = pd.DataFrame({"date": [d.isoformat() for d in found],
                        "cmz_rain": np.round(real_labeller.cmz_rainfall(fields, imd.LATS, imd.LONS), 3)})
    os.makedirs(out_dir, exist_ok=True)
    obs.to_csv(os.path.join(out_dir, "observations.csv"), index=False)
    cmz.to_csv(os.path.join(out_dir, "cmz_rainfall.csv"), index=False)
    logger.info("observations: %d district-days from %d IMD days", len(obs), len(found))
    return obs


def stage_forecast_openmeteo(client, feats, start: date, end: date, model: str, leads, per_district: int,
                             out_dir: str, start_hour: int = 3, offset: int = 0) -> pd.DataFrame:
    from src.data.real.openmeteo import fetch_previous_run_precip
    pts, owner = sample_points(feats, per_district)
    per_point = fetch_previous_run_precip(client, pts, start, end, model, leads, start_hour, offset)
    rows = []
    for did in dict.fromkeys(owner):
        frames = [per_point[i] for i, o in enumerate(owner) if o == did and not per_point[i].empty]
        if not frames:
            continue
        mean = pd.concat(frames).groupby(level=0).mean()
        for d, r in mean.iterrows():
            rows.append({"date": pd.Timestamp(d).date().isoformat(), "district_id": did,
                         **{k: round(float(v), 2) for k, v in r.items()}})
    nwp = pd.DataFrame(rows)
    nwp["nwp_model"] = model
    nwp.to_csv(os.path.join(out_dir, f"forecast_{model}.csv"), index=False)
    logger.info("forecast %s: %d district-days", model, len(nwp))
    return nwp


def _per_district(frames: List[pd.DataFrame], feats: List[dict]) -> pd.DataFrame:
    rows = []
    for ft, df in zip(feats, frames):
        if df is None or df.empty:
            continue
        d = df.copy()
        d.index = [pd.Timestamp(x).date().isoformat() for x in d.index]
        d.index.name = "date"
        d = d.reset_index()
        d["district_id"] = ft["properties"]["district_id"]
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stage_dynamics(frames: List[pd.DataFrame], feats, out_dir: str, source: str) -> pd.DataFrame:
    dyn = _per_district(frames, feats)
    dyn["dynamics_source"] = source
    dyn.to_csv(os.path.join(out_dir, "dynamics.csv"), index=False)
    return dyn


def stage_olr(frames: List[pd.DataFrame], feats, out_dir: str, source: str) -> pd.DataFrame:
    olr = _per_district(frames, feats)
    if olr.empty:
        olr = pd.DataFrame(columns=["date", "district_id", "olr", "olr_anomaly"])
    olr["olr_source"] = source
    olr.to_csv(os.path.join(out_dir, "olr.csv"), index=False)
    return olr


def stage_terrain(elev: Optional[np.ndarray], slope: Optional[np.ndarray], feats, out_dir: str) -> pd.DataFrame:
    rows = []
    for k, ft in enumerate(feats):
        p = ft["properties"]
        rows.append({"district_id": p["district_id"],
                     "elevation": float(elev[k]) if elev is not None else float(p.get("elevation_m", 0.0)),
                     "slope": float(slope[k]) if slope is not None else np.nan,
                     "terrain_source": "Copernicus GLO-90 via Open-Meteo" if elev is not None else "study metadata"})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "terrain.csv"), index=False)
    return t


# ------------------------------------------------------------------ assemble
def _read(out_dir: str, name: str) -> pd.DataFrame:
    p = os.path.join(out_dir, name)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def alignment_diagnostic(df: pd.DataFrame) -> Dict[str, float]:
    """Correlation of obs with Day-1 NWP shifted by -1/0/+1 days: catches IMD-day misalignment."""
    out = {}
    d = df[["district_id", "date", "obs_rain_mean", "raw_nwp_d1"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    for k in (-1, 0, 1):
        s = d.copy()
        s["date"] = s["date"] + pd.Timedelta(days=k)
        m = d[["district_id", "date", "obs_rain_mean"]].merge(
            s[["district_id", "date", "raw_nwp_d1"]], on=["district_id", "date"]).dropna()
        out[f"shift_{k:+d}"] = round(float(np.corrcoef(m["obs_rain_mean"], m["raw_nwp_d1"])[0, 1]), 4) if len(m) > 30 else None
    return out


def assemble(feats: List[dict], interim: str, out_dir: str, split: Dict[str, List[int]],
             label_cfg: Optional[Dict] = None, sources: Optional[Dict[str, str]] = None,
             model: Optional[str] = None, day_offset: int = 0) -> str:
    """
    day_offset: interim forecast / label-dynamics tables are keyed by the window
    (D 03 UTC, D+1 03 UTC]. IMD labels rain by the day the 24 h period *ends* (0830 IST), so with
    day_offset=1 those windows are relabelled D+1 to line up with the observations. The value is
    confirmed empirically by the lag-correlation check written to dataset_metadata.json.
    """
    obs = _read(interim, "observations.csv")
    if obs.empty:
        raise FileNotFoundError(f"{interim}/observations.csv missing - run the observations stage")
    fc_files = sorted(f for f in os.listdir(interim) if f.startswith("forecast_"))
    if model:
        fc_files = [f"forecast_{model}.csv"]
    if not fc_files:
        raise FileNotFoundError("no forecast_*.csv in interim - run the forecast stage")
    nwp = pd.read_csv(os.path.join(interim, fc_files[0]))
    dyn = _read(interim, "dynamics.csv")
    olr = _read(interim, "olr.csv")
    ter = _read(interim, "terrain.csv")
    cmz = _read(interim, "cmz_rainfall.csv")

    if day_offset:
        shift = lambda d: (pd.to_datetime(d) + pd.Timedelta(days=day_offset)).dt.strftime("%Y-%m-%d")  # noqa: E731
        nwp["date"] = shift(nwp["date"])
        if not dyn.empty:
            lab_cols = [c for c in dyn.columns if c.startswith("lab_")]
            lab = dyn[["date", "district_id", *lab_cols]].copy()
            lab["date"] = shift(lab["date"])
            dyn = dyn.drop(columns=lab_cols).merge(lab, on=["date", "district_id"], how="left")

    meta_rows = []
    for ft in feats:
        p = ft["properties"]
        meta_rows.append({"district_id": p["district_id"], "district_name": p["district_name"],
                          "state_name": p["state_name"], "zone": p.get("zone", "Central India"),
                          "latitude": p.get("centroid_lat", p["rep_lat"]),
                          "longitude": p.get("centroid_lon", p["rep_lon"]),
                          "dist_coast": p.get("dist_coast_km", np.nan),
                          "elevation_meta": p.get("elevation_m", np.nan)})
    meta = pd.DataFrame(meta_rows)

    df = obs.merge(nwp, on=["date", "district_id"], how="inner").merge(meta, on="district_id", how="left")
    if not dyn.empty:
        df = df.merge(dyn, on=["date", "district_id"], how="left")
    if not olr.empty:
        df = df.merge(olr, on=["date", "district_id"], how="left")
    if not ter.empty:
        df = df.merge(ter, on="district_id", how="left")
    else:
        df["elevation"] = df["elevation_meta"]
        df["slope"] = np.nan

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    dt = pd.to_datetime(df["date"])
    df["year"], df["month"] = dt.dt.year, dt.dt.month

    # --- predictors dated to issue time: the 00 UTC analysis of D-1 is what a Day-1 forecast
    # for D is issued from, so shift issue-time fields forward one day per district.
    issue_cols = [c for c in ("u850", "v850", "wind_speed_850", "vorticity_850", "q850", "q500",
                              "z500", "mslp", "cape", "olr", "olr_anomaly") if c in df.columns]
    df = df.sort_values(["district_id", "date"]).reset_index(drop=True)
    df[issue_cols] = df.groupby("district_id")[issue_cols].shift(1)

    train = df["year"].isin(split["train"])
    for col, lab in (("mslp", "mslp_anomaly"), ("lab_mslp", "lab_mslp_anomaly"), ("lab_z500", "lab_z500_anomaly"),
                     ("z500", "z500_anomaly")):
        if col in df.columns:
            clim = df[train].groupby(["district_id", "month"])[col].mean().rename("_c")
            df = df.merge(clim, left_on=["district_id", "month"], right_index=True, how="left")
            df[lab] = df[col] - df["_c"]
            df = df.drop(columns="_c")
        else:
            df[lab] = np.nan

    df["moisture_flux"] = df.get("wind_speed_850", np.nan) * df.get("q850", np.nan) * 1000.0
    df["q500"] = df.get("q500", np.nan)

    # Monsoon-trough latitude proxy: latitude of the lowest issue-time MSLP among districts in
    # the 74-88E band (needs >= 5 districts that day); falls back to the seasonal median.
    band = df[(df["longitude"].between(74, 88)) & (df["latitude"].between(15, 32))]
    if "mslp" in df.columns and band["district_id"].nunique() >= 5:
        tl = band.dropna(subset=["mslp"]).loc[lambda x: x.groupby("date")["mslp"].idxmin()].set_index("date")["latitude"]
        df["trough_latitude"] = df["date"].map(tl)
    else:
        df["trough_latitude"] = np.nan

    # --- regime labels from observations + analysis (never from the forecast)
    if not cmz.empty:
        cz = cmz.set_index("date")["cmz_rain"]
        # Climatology from every non-test season of IMD rainfall (the index is an observational
        # quantity; only the held-out test seasons are kept out of it).
        clim = sorted({int(d[:4]) for d in cz.index} - set(split.get("test", [])))
        z = real_labeller.standardised_cmz_index(cz, clim_years=clim)
        cfg = {**real_labeller.DEFAULTS, **(label_cfg or {})}
        sp = real_labeller.spells(z, cfg["cmz_threshold_sd"], cfg["cmz_min_run_days"])
        df["cmz_index"] = df["date"].map(z)
        df["cmz_spell"] = df["date"].map(sp).fillna(0).astype(int)
        # Observed persistence known at issue time: the CMZ index two days before the target day
        # (IMD publishes day D-2's gridded rain on the morning of D-1, before the 00 UTC run).
        lag = lambda s: s.rename(index=lambda d: (pd.Timestamp(d) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))  # noqa: E731
        z2, sp2 = lag(z), lag(sp)
        df["cmz_index_lag2"] = df["date"].map(z2).fillna(0.0)
        df["cmz_spell_lag2"] = df["date"].map(sp2).fillna(0).astype(int)
    else:
        df["cmz_index"], df["cmz_spell"] = np.nan, 0
    for c in ("lab_u850", "lab_v850", "lab_vort850"):
        if c not in df.columns:
            df[c] = np.nan
    df = real_labeller.label(df.fillna({"lab_u850": 0, "lab_v850": 0, "lab_vort850": 0,
                                        "lab_mslp_anomaly": 0, "lab_z500_anomaly": 0}), label_cfg)

    df["obs_heavy"] = (df["obs_rain_max"] >= 64.5).astype(int)
    df["obs_very_heavy"] = (df["obs_rain_max"] >= 115.6).astype(int)
    df["obs_extremely_heavy"] = (df["obs_rain_max"] >= 204.5).astype(int)

    # Predictor gaps (e.g. OLR outage): fill with the training median, record the fraction.
    predictors = ["u850", "v850", "wind_speed_850", "vorticity_850", "mslp_anomaly", "q500", "cape",
                  "olr", "olr_anomaly", "moisture_flux", "trough_latitude", "elevation", "slope", "dist_coast"]
    defaults = {"olr": 240.0, "olr_anomaly": 0.0, "slope": 0.0}
    fill_report = {}
    # A predictor that exists for only part of the record (e.g. NOAA interpolated OLR ends in
    # 2022) would let the model learn a period effect instead of physics: neutralise it everywhere.
    for c in ("olr", "olr_anomaly", "cape", "q500", "vorticity_850"):
        if c in df.columns and df[c].isna().mean() > 0.05:
            logger.warning("%s missing for %.0f%% of rows - set to a constant (not used)", c,
                           100 * df[c].isna().mean())
            fill_report[c + "_neutralised"] = round(float(df[c].isna().mean()), 4)
            df[c] = defaults.get(c, 0.0)
    for c in predictors:
        if c not in df.columns:
            df[c] = np.nan
        frac = float(df[c].isna().mean())
        fill_report[c] = round(frac, 4)
        med = df.loc[train, c].median()
        df[c] = df[c].fillna(defaults.get(c, med if pd.notna(med) else 0.0))

    leads = [c for c in df.columns if c.startswith("raw_nwp_d")]
    before = len(df)
    df = df.dropna(subset=["obs_rain_mean", "raw_nwp_d1"]).reset_index(drop=True)
    for c in leads:
        df[c] = df[c].fillna(df["raw_nwp_d1"])   # rare gaps at longer leads

    cols = ["date", "year", "month", "district_id", "district_name", "state_name", "zone",
            "regime", "regime_name", "regime_rule", "obs_rain_mean", "obs_rain_max", "obs_rain_p90",
            "obs_heavy", "obs_very_heavy", "obs_extremely_heavy", *sorted(leads),
            *predictors, "latitude", "longitude", "cmz_index", "cmz_spell", "cmz_index_lag2", "cmz_spell_lag2",
            "obs_cells", "nwp_model"]
    df = df[[c for c in cols if c in df.columns]]

    years = sorted(df["year"].unique().tolist())
    split = {k: [y for y in v if y in years] for k, v in split.items()}
    if not all(split.values()):
        if len(years) < 3:
            raise ValueError(f"only {len(years)} season(s) with both observations and forecasts {years}; "
                             "need >= 3 for a train/val/test split - add seasons or pick a model with a "
                             "longer archive (scripts/probe_forecast_archive.py)")
        split = {"train": years[:-2], "val": [years[-2]], "test": [years[-1]]}
        logger.info("configured split not covered by the data; using %s", split)

    path = write_table(df, os.path.join(out_dir, "district_daily.parquet"))
    meta_out = {
        "dataset_type": "REAL_OBSERVED_IMD_NWP",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "years": years, "split": split,
        "total_district_records": int(len(df)), "dropped_incomplete_rows": int(before - len(df)),
        "districts_count": int(df["district_id"].nunique()),
        "total_days": int(df["date"].nunique()),
        "sources": sources or {},
        "nwp_model": str(df["nwp_model"].iloc[0]) if "nwp_model" in df.columns and len(df) else None,
        "predictor_timing": "issue-time: 00 UTC analysis of D-1 (available when the Day-1 forecast is issued)",
        "imd_day_convention": {"window": "24 h ending 03 UTC (0830 IST) on the labelled day" if day_offset == 1
                               else "24 h starting 03 UTC on the labelled day", "day_offset": day_offset},
        "regime_labels": "physically defined from IMD obs + analysis (src/regime/real_labeller.py)",
        "regime_counts": {real_labeller.REGIME_NAMES[int(k)]: int(v)
                          for k, v in df["regime"].value_counts().sort_index().items()},
        "predictor_missing_fraction": fill_report,
        "day_alignment_check": alignment_diagnostic(df),
        "provenance_note": ("Observed IMD 0.25 deg gridded rainfall aggregated to Census-2011 district "
                            "polygons; raw forecasts are archived operational NWP runs. Skill figures on "
                            "this archive are genuine out-of-sample verification."),
    }
    al = meta_out["day_alignment_check"]
    if al.get("shift_+0") is not None:
        best = max((v, k) for k, v in al.items() if v is not None)
        if best[1] != "shift_+0" and best[0] > (al["shift_+0"] or 0) + 0.05:
            logger.warning("Day alignment: obs correlate best with Day-1 NWP at %s (%.3f vs %.3f). "
                           "Check observations.day_offset in configs/data_sources.yaml.",
                           best[1], best[0], al["shift_+0"])
    with open(os.path.join(out_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    logger.info("real archive: %d rows, %d districts, years %s -> %s", len(df),
                meta_out["districts_count"], years, path)
    return path
