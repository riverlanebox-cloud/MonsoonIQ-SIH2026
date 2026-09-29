"""
Spatial verification — Fractions Skill Score (FSS).

District verification answers "is the number right where we issued it". FSS
answers the other operational question the statement names: "is the spatial
pattern right, and at what neighbourhood scale does it become useful".

Method, stated precisely so it can be attacked:

  1. The correction models are fitted on district-aggregated observations, because
     that is the unit the product is issued on.
  2. For each grid cell the multiplicative bias adjustment of its district,
     (corrected_mean / raw_mean), is applied to the raw grid value. Cells outside
     every district inherit the adjustment of the nearest district centroid, so
     the whole land domain carries a correction rather than a patchwork.
  3. FSS is then computed between the adjusted grid and the generator's gridded
     truth, for the raw model, a regime-agnostic quantile mapping and the
     regime-aware correction, over 1x1 / 3x3 / 5x5 / 9x9 cell neighbourhoods.

On the synthetic archive both the grid truth and the raw grid forecast are
simulated and these numbers only validate the spatial pipeline. On the real
archive the truth is IMD 0.25 deg rainfall regridded to 0.5 deg and the raw grid
forecast is GFS.
"""

import json
import os
from typing import Dict, Any, List, Tuple

import logging

import numpy as np

from src.verification.metrics import compute_fractions_skill_score_2d
from src.verification.stratified import reliability_class

from src.config import GRID_NPZ  # noqa: E402  (mode-dependent path)
DISTRICT_GEOJSON = ("data/geojson/india_districts_census2011.geojson"
                    if os.path.exists("data/geojson/india_districts_census2011.geojson")
                    else "data/geojson/india_districts.geojson")
THRESHOLDS = (15.6, 64.5)
WINDOWS = (1, 3, 5, 9)
RATIO_CLIP = (0.25, 4.0)
MIN_RAW_MEAN_MM = 0.5
BOOTSTRAP_ITERS = 400
logger = logging.getLogger(__name__)


def _grid_provenance() -> str:
    from src import config
    return ("Grid truth is IMD 0.25 deg rainfall regridded to 0.5 deg; the raw grid forecast is GFS."
            if config.MODE == "real" else "Grid truth and raw grid forecast are synthetic.")


def _cell_district_map(npz, geojson_path: str) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """District index for every land cell: containment first, else nearest centroid."""
    import json as _json
    from src.compat import points_in_polygon

    with open(geojson_path, "r", encoding="utf-8") as f:
        geo = _json.load(f)

    ids, geoms, centroids, bounds = [], [], [], []
    for feat in geo["features"]:
        props = feat["properties"]
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        rings = [np.asarray(r, dtype=float) for poly in polys for r in poly]
        allp = np.vstack(rings)
        ids.append(props["district_id"])
        geoms.append(rings)
        centroids.append((float(props["centroid_lon"]), float(props["centroid_lat"])))
        bounds.append((allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()))

    lats = npz["lats"].astype(float)
    lons = npz["lons"].astype(float)
    land_index = npz["land_index"].astype(int)
    shape_grid = tuple(npz["grid_shape"].astype(int))
    rows, cols = np.unravel_index(land_index, shape_grid)
    cell_lon = lons[cols]
    cell_lat = lats[rows]

    n = land_index.size
    assign = np.full(n, -1, dtype=int)
    for i, geom in enumerate(geoms):
        minx, miny, maxx, maxy = bounds[i]
        cand = ((cell_lon >= minx) & (cell_lon <= maxx)
                & (cell_lat >= miny) & (cell_lat <= maxy))
        if not cand.any():
            continue
        inside = points_in_polygon(geom, cell_lon[cand], cell_lat[cand])
        assign[np.flatnonzero(cand)[inside]] = i

    contained = int((assign >= 0).sum())
    if contained < n:
        cents = np.array(centroids)
        missing = np.flatnonzero(assign < 0)
        d2 = ((cell_lon[missing, None] - cents[None, :, 0]) ** 2
              + (cell_lat[missing, None] - cents[None, :, 1]) ** 2)
        assign[missing] = np.argmin(d2, axis=1)

    meta = {
        "land_cells": int(n),
        "cells_inside_district_polygons": contained,
        "cells_assigned_to_nearest_district": int(n - contained),
        "note": (("Census-2011 district boundaries; cells outside all 53 districts are assigned "
                  "to the nearest district centroid.") if "census2011" in geojson_path else
                 ("District boundaries in this archive are simplified boxes, so most cells "
                  "are assigned by nearest-centroid rather than containment.")),
    }
    return assign, ids, meta


def _factor_field(assign: np.ndarray, district_ids: List[str],
                  raw_means: Dict[str, float],
                  corrected_means: Dict[str, float]) -> np.ndarray:
    factors = np.ones(len(district_ids), dtype=float)
    for i, did in enumerate(district_ids):
        r = raw_means.get(did)
        c = corrected_means.get(did)
        if r is None or c is None or r < MIN_RAW_MEAN_MM:
            continue
        factors[i] = float(np.clip(c / r, *RATIO_CLIP))
    return factors[assign]


def run_grid_verification(moe, df, test_df, thresholds=THRESHOLDS, windows=WINDOWS,
                          grid_model=None,
                          npz_path: str = GRID_NPZ,
                          geojson_path: str = DISTRICT_GEOJSON,
                          nwp_col: str = "raw_nwp_d1",
                          seed: int = 20260922) -> Dict[str, Any]:
    if not os.path.exists(npz_path):
        return {"available": False,
                "reason": f"{npz_path} missing - run scripts/build_grid_samples.py"}

    data = np.load(npz_path, allow_pickle=True)
    meta_keys = {"lats", "lons", "land_index", "grid_shape", "elevation_grid", "land_mask"}
    date_keys = sorted(k for k in data.keys() if k not in meta_keys)
    if not date_keys:
        return {"available": False, "reason": "no date records in grid archive"}

    assign, district_ids, cell_meta = _cell_district_map(data, geojson_path)
    grid_shape = tuple(data["grid_shape"].astype(int))
    test_dates = set(test_df["date"].astype(str).unique())

    from src.regime.ml_classifier import MLRegimeClassifier
    clf = MLRegimeClassifier()
    clf.load()

    # Optional grid-native correction (fitted on grid cells, not district means).
    native_fields: Dict[str, np.ndarray] = {}
    if grid_model is None:
        try:
            from src.correction.grid_correction import GridCorrectionModel
            grid_model = GridCorrectionModel.load()
            logger.info("Loaded grid-native correction for FSS")
        except Exception:
            grid_model = None
    if grid_model is not None:
        try:
            native_fields = {k: v["grid_correction"] for k, v in
                             grid_model.verification_fields(clf).items()}
        except Exception as exc:  # pragma: no cover
            logger.warning("Grid-native fields unavailable: %s", exc)
            native_fields = {}

    per_date = []
    for key in date_keys:
        if key not in test_dates:
            continue
        day_df = df[df["date"] == key]
        if day_df.empty:
            continue
        day_df = day_df.reset_index(drop=True)
        posteriors = clf.predict_proba(day_df)
        preds = moe.predict_all_systems(day_df, nwp_col=nwp_col, regime_probs=posteriors)

        raw_means = dict(zip(day_df["district_id"], day_df["raw_nwp_d1"].astype(float)))
        moe_means = dict(zip(day_df["district_id"], preds["monsooniq"].astype(float)))
        qm_means = dict(zip(day_df["district_id"], preds["global_qm"].astype(float)))

        rec = data[key].item()
        truth_flat = rec["true_rain"].astype(float)
        raw_flat = rec["raw_nwp_d1"].astype(float)

        fields = {}
        for name, means in (("raw_nwp", None),
                            ("global_qm_district_transfer", qm_means),
                            ("regime_aware_district_transfer", moe_means)):
            if means is None:
                fields[name] = raw_flat
            else:
                fields[name] = np.maximum(
                    0.0, raw_flat * _factor_field(assign, district_ids, raw_means, means))
        if key in native_fields:
            fields["regime_aware_grid_native"] = np.maximum(0.0, native_fields[key])

        # Rebuild 2-D fields (non-land cells stay zero in both truth and forecast).
        def to_grid(flat):
            grid = np.zeros(grid_shape, dtype=float)
            grid.ravel()[data["land_index"].astype(int)] = flat
            return grid

        per_date.append({
            "date": key,
            "truth": to_grid(truth_flat),
            "fields": {k: to_grid(v) for k, v in fields.items()},
            "heavy_cells": int((truth_flat >= 64.5).sum()),
        })

    if not per_date:
        return {"available": False,
                "reason": "no gridded sample dates fall inside the verification period"}

    system_names = ["raw_nwp", "global_qm_district_transfer",
                    "regime_aware_district_transfer"]
    if any("regime_aware_grid_native" in e["fields"] for e in per_date):
        system_names.append("regime_aware_grid_native")

    scores: Dict[str, Any] = {}
    for thr in thresholds:
        for window in windows:
            for system in system_names:
                vals = []
                for e in per_date:
                    try:
                        vals.append(compute_fractions_skill_score_2d(
                            e["fields"][system], e["truth"], thr, window))
                    except Exception:
                        continue
                if not vals:
                    continue
                vals = np.array(vals, dtype=float)
                rng = np.random.RandomState(seed)
                boot = np.array([vals[rng.randint(0, vals.size, vals.size)].mean()
                                 for _ in range(BOOTSTRAP_ITERS)])
                entry = scores.setdefault(f"{thr}", {}).setdefault(f"w{window}", {})
                entry[system] = {
                    "fss": round(float(vals.mean()), 4),
                    "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                             round(float(np.percentile(boot, 97.5)), 4)],
                }
                if system != "raw_nwp":
                    raw_mean = np.mean([
                        compute_fractions_skill_score_2d(e["fields"]["raw_nwp"], e["truth"], thr, window)
                        for e in per_date])
                    entry["delta_vs_raw_nwp"] = round(float(vals.mean() - raw_mean), 4)

        total_events = int(sum(int((e["truth"] >= thr).sum()) for e in per_date))
        per_date_mean = int(round(total_events / len(per_date)))
        scores[f"{thr}"]["event_bookkeeping"] = {
            "grid_dates_scored": int(len(per_date)),
            "land_cells_per_date": int(assign.size),
            "total_observed_cells_above_threshold": total_events,
            "dates_with_any_event": int(sum(1 for e in per_date if bool((e["truth"] >= thr).any()))),
            "mean_cells_per_date_above_threshold": round(total_events / len(per_date), 1),
            "reliability": reliability_class(per_date_mean),
        }

    return {
        "available": True,
        "method": ("Two grid paths are scored. `*_district_transfer` extends district-scale "
                   "multiplicative adjustments to cells by nearest-district assignment; "
                   "`regime_aware_grid_native` is a correction fitted on grid cells. "
                   + _grid_provenance()),
        "systems": system_names,
        "grid": {**cell_meta, "grid_shape": [int(x) for x in grid_shape],
                 "dates_scored": int(len(per_date)),
                 "dates_in_verification_period": int(len(test_dates))},
        "thresholds_mm": list(thresholds),
        "neighbourhood_windows_cells": list(windows),
        "scores": scores,
    }


if __name__ == "__main__":
    import pandas as pd
    from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts

    from src import config
    from src.compat import read_table
    df = read_table(config.ARCHIVE)
    test_df = df[df["year"].isin(config.TEST_YEARS)].reset_index(drop=True)
    print(json.dumps(run_grid_verification(MonsoonIQMixtureOfExperts.load(), df, test_df),
                     indent=2, default=float)[:5000])
