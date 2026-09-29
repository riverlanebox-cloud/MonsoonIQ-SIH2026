"""
Open-Meteo client for archived NWP forecasts, analysis-like dynamics and terrain.

Why Open-Meteo: it is the only free, key-less archive of *what operational global models
actually forecast N days ahead* (Previous Runs API), covering NCEP GFS, ECMWF IFS and the
UK Met Office Unified Model - the model family NCMRWF's NCUM-G is built on. For the
operational NCUM-G / NEPS-G archive use src/data/real/gridded_nwp.py instead.

APIs used
  Previous Runs      https://previous-runs-api.open-meteo.com/v1/forecast
                     hourly <var>_previous_dayN = value predicted 24*N h before valid time
  Historical Forecast https://historical-forecast-api.open-meteo.com/v1/forecast
                     analysis-like series stitched from the first hours of each run;
                     has pressure-level fields (850/500 hPa)
  Elevation          https://api.open-meteo.com/v1/elevation  (Copernicus GLO-90 DEM)

Free non-commercial limits are ~10,000 calls/day; a call with > 10 variables or > 2 weeks
is weighted as several. Every response is cached on disk, so a fetch interrupted by the daily
limit resumes where it stopped when re-run the next day.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

DYNAMIC_VARS = [
    "wind_speed_850hPa", "wind_direction_850hPa", "relative_humidity_850hPa", "temperature_850hPa",
    "relative_humidity_500hPa", "temperature_500hPa", "geopotential_height_500hPa",
    "pressure_msl", "cape",
]
STENCIL_VARS = ["wind_speed_850hPa", "wind_direction_850hPa"]


class DailyLimitReached(RuntimeError):
    """Raised when Open-Meteo reports the daily quota is used up; re-run tomorrow to resume."""


class OpenMeteoClient:
    def __init__(self, cache_dir: str = "data/real/raw/openmeteo_cache", min_interval_s: float = 0.25,
                 api_key: Optional[str] = None):
        self.cache_dir = cache_dir
        self.min_interval_s = min_interval_s
        self.api_key = api_key or os.getenv("OPEN_METEO_API_KEY")   # commercial customer-* hosts
        self._last = 0.0
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, url: str, params: dict) -> str:
        blob = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        return os.path.join(self.cache_dir, hashlib.sha1(blob.encode()).hexdigest() + ".json")

    def get(self, url: str, params: dict, retries: int = 6):
        path = self._key(url, params)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        import requests
        q = dict(params)
        if self.api_key:
            q["apikey"] = self.api_key
        for attempt in range(retries):
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = requests.get(url, params=q, timeout=120)
            except Exception as exc:
                logger.warning("Open-Meteo request failed (%s); retrying", exc)
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                with open(path + ".part", "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(path + ".part", path)
                return data
            reason = ""
            try:
                reason = r.json().get("reason", "")
            except Exception:
                reason = r.text[:200]
            if r.status_code == 429 or "limit" in reason.lower():
                if "daily" in reason.lower():
                    raise DailyLimitReached(reason)
                logger.info("Open-Meteo rate limit (%s); sleeping 65 s", reason)
                time.sleep(65)
                continue
            if r.status_code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"Open-Meteo {r.status_code}: {reason} ({url})")
        raise RuntimeError(f"Open-Meteo request kept failing: {url}")


def _chunks(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _date_chunks(start: date, end: date, days: int = 31):
    d = start
    while d <= end:
        e = min(end, d + timedelta(days=days - 1))
        yield d, e
        d = e + timedelta(days=1)


def fetch_hourly(client: OpenMeteoClient, url: str, points: List[Tuple[float, float]],
                 start: date, end: date, hourly: List[str], extra: Optional[dict] = None,
                 batch: int = 50) -> List[pd.DataFrame]:
    """
    Hourly series (UTC) for each point. Returns one DataFrame per point indexed by time,
    in the order of `points`. Multi-location requests keep the call count low.
    """
    frames: List[List[pd.DataFrame]] = [[] for _ in points]
    idx = list(range(len(points)))
    for s, e in _date_chunks(start, end):
        for group in _chunks(idx, batch):
            params = {
                "latitude": ",".join(f"{points[i][0]:.4f}" for i in group),
                "longitude": ",".join(f"{points[i][1]:.4f}" for i in group),
                "hourly": ",".join(hourly),
                "start_date": s.isoformat(), "end_date": e.isoformat(),
                "timezone": "GMT", "wind_speed_unit": "ms", "precipitation_unit": "mm",
                "cell_selection": "land",
            }
            if extra:
                params.update(extra)
            data = client.get(url, params)
            items = data if isinstance(data, list) else [data]
            for i, item in zip(group, items):
                h = item.get("hourly", {})
                if not h or "time" not in h:
                    continue
                df = pd.DataFrame(h)
                df["time"] = pd.to_datetime(df["time"], utc=True)
                frames[i].append(df.set_index("time"))
    return [pd.concat(f).sort_index() if f else pd.DataFrame() for f in frames]


def imd_day_key(times: pd.DatetimeIndex, start_hour_utc: int = 3, offset_days: int = 0) -> pd.Series:
    """
    Map hourly *accumulation* timestamps (value at T = amount in (T-1h, T]) onto the IMD
    rainfall day: date D covers (D start_hour, D+1 start_hour] UTC.
    """
    shifted = times - pd.Timedelta(hours=start_hour_utc + 1)
    return pd.Series((shifted.floor("D") + pd.Timedelta(days=offset_days)).date, index=times)


def daily_accumulation(hourly: pd.Series, start_hour_utc: int = 3, offset_days: int = 0,
                       min_hours: int = 22) -> pd.Series:
    if hourly.empty:
        return pd.Series(dtype=float)
    key = imd_day_key(hourly.index, start_hour_utc, offset_days)
    g = hourly.groupby(key.values)
    out = g.sum(min_count=1)
    out[g.count() < min_hours] = np.nan
    out.index = pd.Index(out.index, name="date")
    return out


def fetch_previous_run_precip(client: OpenMeteoClient, points: List[Tuple[float, float]],
                              start: date, end: date, model: str, leads: Sequence[int] = (1, 2, 3, 4, 5),
                              start_hour_utc: int = 3, offset_days: int = 0) -> List[pd.DataFrame]:
    """
    Daily (IMD-window) precipitation forecast at each lead for each point.
    Returns per point a DataFrame indexed by date with columns raw_nwp_d1..d5 (mm/day).
    Each hourly value tagged _previous_dayN comes from the run initialised >= 24*N h
    before that hour, so the Day-N column is a genuine N-day-ahead forecast.
    """
    hourly_vars = [f"precipitation_previous_day{n}" for n in leads]
    # widen the window so the first/last IMD days are complete
    series = fetch_hourly(client, PREVIOUS_RUNS_URL, points, start - timedelta(days=1),
                          end + timedelta(days=1), hourly_vars, extra={"models": model})
    out = []
    for df in series:
        cols = {}
        for n, v in zip(leads, hourly_vars):
            if v in df:
                cols[f"raw_nwp_d{n}"] = daily_accumulation(df[v].astype(float), start_hour_utc, offset_days)
        daily = pd.DataFrame(cols)
        if not daily.empty:
            daily = daily[(daily.index >= start) & (daily.index <= end)]
        out.append(daily)
    return out


def _uv(speed: np.ndarray, direction_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Meteorological direction (where wind comes FROM) -> u (eastward), v (northward)."""
    rad = np.radians(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def specific_humidity(rh_pct: np.ndarray, temp_c: np.ndarray, p_hpa: float) -> np.ndarray:
    """q (kg/kg) from RH and T via Bolton (1980) saturation vapour pressure."""
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    e = np.clip(rh_pct, 0, 100) / 100.0 * es
    return 0.622 * e / (p_hpa - 0.378 * e)


def fetch_dynamics(client: OpenMeteoClient, points: List[Tuple[float, float]], start: date, end: date,
                   model: str = "gfs_seamless", stencil_deg: float = 0.5,
                   analysis_hour_utc: int = 0, stencil: str = "forward") -> List[pd.DataFrame]:
    """
    Issue-time analysis for each point: instantaneous values at `analysis_hour_utc` of each
    date (00 UTC = the analysis a Day-1 forecast is issued from), plus the 24-h mean of the
    same fields over the IMD day for regime *labelling*.

    Columns per date:
      u850 v850 wind_speed_850 vorticity_850 q850 q500 z500 mslp cape    (issue-time, 00 UTC)
      lab_u850 lab_v850 lab_vort850 lab_mslp lab_z500                    (IMD-day mean, labels)
    Relative vorticity: stencil="cross" uses a centred difference on a +/- stencil_deg cross
    (4 extra points per district); "forward" uses the centre plus an east and a north point
    (2 extra points - half the API quota, first-order accurate, ample for a threshold label).
    """
    centre = fetch_hourly(client, HISTORICAL_FORECAST_URL, points, start - timedelta(days=1),
                          end + timedelta(days=1), DYNAMIC_VARS, extra={"models": model})
    d = stencil_deg
    per = 4 if stencil == "cross" else 2
    cross = []
    for lat, lon in points:
        cross += ([(lat, lon + d), (lat, lon - d), (lat + d, lon), (lat - d, lon)] if per == 4
                  else [(lat, lon + d), (lat + d, lon)])
    stencil_frames = fetch_hourly(client, HISTORICAL_FORECAST_URL, cross, start - timedelta(days=1),
                                  end + timedelta(days=1), STENCIL_VARS, extra={"models": model})

    out = []
    for k, (lat, lon) in enumerate(points):
        c = centre[k]
        if c.empty:
            out.append(pd.DataFrame())
            continue
        u, v = _uv(c["wind_speed_850hPa"].to_numpy(float), c["wind_direction_850hPa"].to_numpy(float))
        f = pd.DataFrame(index=c.index)
        f["u850"], f["v850"] = u, v
        f["wind_speed_850"] = c["wind_speed_850hPa"].astype(float)
        f["q850"] = specific_humidity(c["relative_humidity_850hPa"].to_numpy(float),
                                      c["temperature_850hPa"].to_numpy(float), 850.0)
        f["q500"] = specific_humidity(c["relative_humidity_500hPa"].to_numpy(float),
                                      c["temperature_500hPa"].to_numpy(float), 500.0)
        f["z500"] = c["geopotential_height_500hPa"].astype(float)
        f["mslp"] = c["pressure_msl"].astype(float)
        f["cape"] = c["cape"].astype(float)

        if per == 4:
            e, w, n, s = stencil_frames[4 * k: 4 * k + 4]
            span = 2 * d
        else:
            e, n = stencil_frames[2 * k: 2 * k + 2]
            w = s = c          # forward difference against the centre point
            span = d
        if all(not x.empty for x in (e, w, n, s)):
            ue, ve = _uv(e["wind_speed_850hPa"].to_numpy(float), e["wind_direction_850hPa"].to_numpy(float))
            uw, vw = _uv(w["wind_speed_850hPa"].to_numpy(float), w["wind_direction_850hPa"].to_numpy(float))
            un, vn = _uv(n["wind_speed_850hPa"].to_numpy(float), n["wind_direction_850hPa"].to_numpy(float))
            us, vs = _uv(s["wind_speed_850hPa"].to_numpy(float), s["wind_direction_850hPa"].to_numpy(float))
            dx = span * 111_000.0 * np.cos(np.radians(lat))
            dy = span * 111_000.0
            m = min(len(ue), len(uw), len(un), len(us), len(f))
            vort = np.full(len(f), np.nan)
            vort[:m] = (ve[:m] - vw[:m]) / dx - (un[:m] - us[:m]) / dy
            f["vorticity_850"] = vort
        else:
            f["vorticity_850"] = np.nan

        issue = f[f.index.hour == analysis_hour_utc].copy()
        issue.index = pd.Index(issue.index.date, name="date")
        lab = f[["u850", "v850", "vorticity_850", "mslp", "z500"]].copy()
        key = imd_day_key(lab.index + pd.Timedelta(hours=1))   # instantaneous: no accumulation shift
        lab_daily = lab.groupby(key.values).mean()
        lab_daily.columns = ["lab_u850", "lab_v850", "lab_vort850", "lab_mslp", "lab_z500"]
        lab_daily.index = pd.Index(lab_daily.index, name="date")
        daily = issue.join(lab_daily, how="outer")
        daily = daily[(daily.index >= start) & (daily.index <= end)]
        out.append(daily)
    return out


def fetch_elevation(client: OpenMeteoClient, points: List[Tuple[float, float]]) -> np.ndarray:
    vals: List[float] = []
    for group in _chunks(points, 100):
        data = client.get(ELEVATION_URL, {
            "latitude": ",".join(f"{p[0]:.4f}" for p in group),
            "longitude": ",".join(f"{p[1]:.4f}" for p in group),
        })
        vals += [float(x) for x in data.get("elevation", [np.nan] * len(group))]
    return np.asarray(vals, dtype=float)


def terrain(client: OpenMeteoClient, points: List[Tuple[float, float]], stencil_deg: float = 0.1
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Elevation (m) at each point and slope magnitude (m/m) from a +/- stencil."""
    d = stencil_deg
    pts = []
    for lat, lon in points:
        pts += [(lat, lon), (lat, lon + d), (lat, lon - d), (lat + d, lon), (lat - d, lon)]
    z = fetch_elevation(client, pts).reshape(-1, 5)
    dx = 2 * d * 111_000.0 * np.cos(np.radians([p[0] for p in points]))
    dy = 2 * d * 111_000.0
    slope = np.hypot((z[:, 1] - z[:, 2]) / dx, (z[:, 3] - z[:, 4]) / dy)
    return np.maximum(z[:, 0], 0.0), slope
