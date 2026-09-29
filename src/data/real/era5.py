"""
ERA5 reanalysis dynamics via the Copernicus Climate Data Store (free account + key).

Hersbach et al. (2020), QJRMS 146:1999-2049. 0.25 deg, hourly; we take the 00 UTC analysis
(issue time) and the 00/06/12/18 UTC mean over the IMD day (labels). Variables:
  pressure levels 850/500 hPa : u, v, vo (relative vorticity), q, z
  single levels               : mean_sea_level_pressure, convective_available_potential_energy

Requires `pip install cdsapi netCDF4 xarray` and ~/.cdsapirc:
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
Output columns match src/data/real/openmeteo.fetch_dynamics, so build_archive treats both alike.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
AREA = [38.5, 66.5, 6.5, 100.0]   # N, W, S, E


def download(years: List[int], months: List[int], out_dir: str) -> List[str]:
    import cdsapi
    c = cdsapi.Client()
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for y in years:
        pl = os.path.join(out_dir, f"era5_pl_{y}.nc")
        sl = os.path.join(out_dir, f"era5_sl_{y}.nc")
        common = {"product_type": ["reanalysis"], "year": [str(y)], "month": [f"{m:02d}" for m in months],
                  "day": [f"{d:02d}" for d in range(1, 32)], "time": ["00:00", "06:00", "12:00", "18:00"],
                  "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}
        if not os.path.exists(pl):
            c.retrieve("reanalysis-era5-pressure-levels", {**common, "pressure_level": ["500", "850"],
                       "variable": ["u_component_of_wind", "v_component_of_wind", "vorticity",
                                    "specific_humidity", "geopotential"]}, pl)
        if not os.path.exists(sl):
            c.retrieve("reanalysis-era5-single-levels", {**common, "variable": [
                "mean_sea_level_pressure", "convective_available_potential_energy"]}, sl)
        paths += [pl, sl]
    return paths


def sample_points(out_dir: str, years: List[int], points: List[Tuple[float, float]],
                  start: date, end: date) -> List[pd.DataFrame]:
    import xarray as xr
    frames = [[] for _ in points]
    for y in years:
        pl = xr.open_dataset(os.path.join(out_dir, f"era5_pl_{y}.nc"))
        sl = xr.open_dataset(os.path.join(out_dir, f"era5_sl_{y}.nc"))
        tn = "valid_time" if "valid_time" in pl.dims else "time"
        lev = "pressure_level" if "pressure_level" in pl.dims else "level"
        for k, (lat, lon) in enumerate(points):
            p = pl.sel(latitude=lat, longitude=lon, method="nearest")
            s = sl.sel(latitude=lat, longitude=lon, method="nearest")
            f = pd.DataFrame({
                "u850": p["u"].sel({lev: 850}).values, "v850": p["v"].sel({lev: 850}).values,
                "vorticity_850": p["vo"].sel({lev: 850}).values,
                "q850": p["q"].sel({lev: 850}).values, "q500": p["q"].sel({lev: 500}).values,
                "z500": p["z"].sel({lev: 500}).values / 9.80665,
                "mslp": s["msl"].values / 100.0, "cape": s["cape"].values,
            }, index=pd.to_datetime(p[tn].values, utc=True))
            f["wind_speed_850"] = np.hypot(f["u850"], f["v850"])
            frames[k].append(f)
        pl.close()
        sl.close()
    out = []
    for fr in frames:
        f = pd.concat(fr).sort_index()
        issue = f[f.index.hour == 0].copy()
        issue.index = pd.Index(issue.index.date, name="date")
        lab = f[["u850", "v850", "vorticity_850", "mslp", "z500"]]
        key = (lab.index - pd.Timedelta(hours=3)).floor("D").date
        lab = lab.groupby(key).mean()
        lab.columns = ["lab_u850", "lab_v850", "lab_vort850", "lab_mslp", "lab_z500"]
        lab.index = pd.Index(lab.index, name="date")
        d = issue.join(lab, how="outer")
        out.append(d[(d.index >= start) & (d.index <= end)])
    return out
