"""
Adapter for operational gridded forecasts - NCMRWF NCUM-G / NEPS-G, IMD GFS, NCEP GFS, ECMWF.

This is the path an NCMRWF deployment would use: the forecast arrives as NetCDF (or GRIB via
cfgrib) on the model grid, and MonsoonIQ needs district-mean daily rainfall at each lead,
accumulated over the IMD day. Two layouts are handled:

  A. one file per initialisation, variable dims (time, lat, lon) with `time` = valid time and
     either daily totals or a running accumulation since init (set cumulative=True);
     the init date is parsed from the file name (first YYYYMMDD in it) or the `init` attribute.
  B. one file with dims (init|time, step|lead, lat, lon), step as timedelta or integer days.

Output: DataFrame [date, district_id, raw_nwp_d1..dN] ready for build_archive.
"""

from __future__ import annotations

import glob
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data import geometry as G

logger = logging.getLogger(__name__)


def _coord(ds, names):
    for n in names:
        if n in ds.coords or n in ds.dims:
            return n
    raise KeyError(f"none of {names} in dataset")


def district_weights(features: List[dict], lats: np.ndarray, lons: np.ndarray) -> Dict[str, tuple]:
    res = float(abs(lats[1] - lats[0]))
    return {f["properties"]["district_id"]: G.cell_weights(f["geometry"], lats, lons, res)
            for f in features}


def _district_means(field: np.ndarray, weights: Dict[str, tuple]) -> Dict[str, float]:
    out = {}
    for did, (idx, w) in weights.items():
        vals = np.array([field[i, j] for i, j in idx], dtype=float)
        ok = ~np.isnan(vals)
        out[did] = float(np.sum(vals[ok] * w[ok]) / w[ok].sum()) if ok.any() else np.nan
    return out


def load_gridded_forecasts(pattern: str, var: str, features: List[dict], leads=(1, 2, 3, 4, 5),
                           cumulative: bool = False, units_per_mm: float = 1.0,
                           day_start_hour_utc: int = 3) -> pd.DataFrame:
    import xarray as xr

    rows: Dict[tuple, Dict[str, float]] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no gridded forecast files match {pattern}")
    weights = None
    for path in files:
        ds = xr.open_dataset(path)
        latn = _coord(ds, ["lat", "latitude", "y"])
        lonn = _coord(ds, ["lon", "longitude", "x"])
        da = ds[var]
        lats = ds[latn].values.astype(float)
        lons = ds[lonn].values.astype(float)
        if lats[0] > lats[-1]:
            da = da.isel({latn: slice(None, None, -1)})
            lats = lats[::-1]
        if weights is None:
            weights = district_weights(features, lats, lons)

        stepn = next((n for n in ("step", "lead", "lead_time", "fcst_day") if n in da.dims), None)
        if stepn is None:   # layout A: valid-time axis, init from name/attr
            m = re.search(r"(20\d{6})", path)
            init = pd.Timestamp(ds.attrs.get("init") or datetime.strptime(m.group(1), "%Y%m%d"))
            tn = _coord(ds, ["time", "valid_time"])
            series = da.values.astype(float) / units_per_mm
            valid = pd.to_datetime(ds[tn].values)
            if cumulative:
                series = np.diff(np.concatenate([np.zeros_like(series[:1]), series]), axis=0)
            # sum sub-daily steps into IMD days, then assign lead = day - init date
            day = (valid - pd.Timedelta(hours=day_start_hour_utc + 1)).floor("D")
            for d in sorted(set(day)):
                lead = (d.normalize() - init.normalize()).days
                if lead not in leads:
                    continue
                field = np.nansum(series[np.asarray(day == d)], axis=0)
                for did, v in _district_means(field, weights).items():
                    rows.setdefault((d.date(), did), {})[f"raw_nwp_d{lead}"] = v
        else:               # layout B
            initn = _coord(ds, ["init", "time", "forecast_reference_time"])
            steps = ds[stepn].values
            step_days = (steps / np.timedelta64(1, "D")).astype(int) if np.issubdtype(steps.dtype, np.timedelta64) \
                else steps.astype(int)
            for ii, init in enumerate(pd.to_datetime(ds[initn].values)):
                for si, lead in enumerate(step_days):
                    if lead not in leads:
                        continue
                    field = da.isel({initn: ii, stepn: si}).values.astype(float) / units_per_mm
                    d = (init.normalize() + timedelta(days=int(lead))).date()
                    for did, v in _district_means(field, weights).items():
                        rows.setdefault((d, did), {})[f"raw_nwp_d{lead}"] = v
        ds.close()
    df = pd.DataFrame([{"date": k[0], "district_id": k[1], **v} for k, v in rows.items()])
    logger.info("gridded forecasts: %d files -> %d district-days", len(files), len(df))
    return df
