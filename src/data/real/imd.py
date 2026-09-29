"""
IMD 0.25 deg daily gridded rainfall - the observational truth for India.

Dataset: Pai D.S. et al. (2014) "Development of a new high spatial resolution (0.25 x 0.25 deg)
long period (1901-2010) daily gridded rainfall data set over India", Mausam 65(1):1-18.
Extended yearly by IMD Pune; ~6,900 rain-gauge stations, inverse-distance interpolation.

Binary layout (same as imdlib, the de-facto reader):
    float32, little-endian, C order, shape (days, 129 lat, 135 lon)
    lat 6.5 .. 38.5 N, lon 66.5 .. 100.0 E, step 0.25, missing = -999.0
Download (no login):
    yearly    POST https://imdpune.gov.in/cmpg/Griddata/rainfall.php         data={"rain": YYYY}
    real-time POST https://imdpune.gov.in/cmpg/Realtimedata/Rainfall/rain.php data={"rain": DDMMYYYY}
"""

from __future__ import annotations

import calendar
import logging
import os
import time
from datetime import date, timedelta
from typing import Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

NLAT, NLON = 129, 135
LATS = np.linspace(6.5, 38.5, NLAT)
LONS = np.linspace(66.5, 100.0, NLON)
MISSING = -999.0
RES = 0.25

YEARLY_URL = "https://imdpune.gov.in/cmpg/Griddata/rainfall.php"
REALTIME_URL = "https://imdpune.gov.in/cmpg/Realtimedata/Rainfall/rain.php"


def yearly_path(raw_dir: str, year: int) -> str:
    return os.path.join(raw_dir, f"Rainfall_ind{year}_rfp25.grd")


def realtime_path(raw_dir: str, day: date) -> str:
    return os.path.join(raw_dir, "realtime", f"rain_ind0.25_{day.strftime('%y_%m_%d')}.grd")


def _post(url: str, data: dict, dest: str, min_bytes: int, retries: int = 4) -> bool:
    import requests
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.post(url, data=data, timeout=180)
            r.raise_for_status()
            if len(r.content) < min_bytes:
                logger.warning("IMD returned %d bytes for %s (expected >= %d) - not yet published?",
                               len(r.content), data, min_bytes)
                return False
            with open(dest + ".part", "wb") as f:
                f.write(r.content)
            os.replace(dest + ".part", dest)
            return True
        except Exception as exc:  # network hiccups on the IMD server are common
            wait = 5 * 2 ** attempt
            logger.warning("IMD download %s failed (%s); retry in %ds", data, exc, wait)
            time.sleep(wait)
    return False


def download_year(year: int, raw_dir: str) -> Optional[str]:
    """Fetch one year of the final gridded product (~25 MB). Skips if present."""
    dest = yearly_path(raw_dir, year)
    days = 366 if calendar.isleap(year) else 365
    expected = days * NLAT * NLON * 4
    if os.path.exists(dest) and os.path.getsize(dest) == expected:
        return dest
    logger.info("Downloading IMD gridded rainfall %d ...", year)
    ok = _post(YEARLY_URL, {"rain": year}, dest, min_bytes=expected // 2)
    return dest if ok else None


def download_realtime(days: Iterable[date], raw_dir: str) -> List[str]:
    """Real-time (provisional) daily files, for seasons IMD has not yet finalised."""
    out = []
    for d in days:
        dest = realtime_path(raw_dir, d)
        if os.path.exists(dest) and os.path.getsize(dest) == NLAT * NLON * 4:
            out.append(dest)
            continue
        if _post(REALTIME_URL, {"rain": d.strftime("%d%m%Y")}, dest, min_bytes=NLAT * NLON * 4):
            out.append(dest)
    return out


def read_grd(path: str, n_days: Optional[int] = None) -> np.ndarray:
    """Read an IMD rainfall .grd into (days, 129, 135) float32 with NaN for missing."""
    raw = np.fromfile(path, dtype="<f4")
    per_day = NLAT * NLON
    if raw.size % per_day != 0:
        raise ValueError(f"{path}: {raw.size} values is not a multiple of {per_day} (129x135)")
    days = raw.size // per_day
    if n_days is not None and days != n_days:
        raise ValueError(f"{path}: {days} days, expected {n_days}")
    arr = raw.reshape(days, NLAT, NLON).astype(np.float32)
    arr[arr <= MISSING + 1] = np.nan
    arr[arr < 0] = np.nan
    return arr


def write_grd(path: str, arr: np.ndarray) -> None:
    """Inverse of read_grd (used by tests and by the real-time stitcher)."""
    out = np.where(np.isnan(arr), MISSING, arr).astype("<f4")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.tofile(path)


def load_days(raw_dir: str, days: List[date]) -> Tuple[np.ndarray, List[date]]:
    """
    Rainfall for the requested days from yearly files, falling back to real-time daily
    files. Returns (array[n, 129, 135], days_found). Days without data are dropped.
    """
    by_year = {}
    for d in days:
        by_year.setdefault(d.year, []).append(d)
    fields, found = [], []
    for year, ds in sorted(by_year.items()):
        path = yearly_path(raw_dir, year)
        if os.path.exists(path):
            n = 366 if calendar.isleap(year) else 365
            arr = read_grd(path, n_days=n)
            start = date(year, 1, 1)
            for d in ds:
                fields.append(arr[(d - start).days])
                found.append(d)
            continue
        for d in ds:
            rt = realtime_path(raw_dir, d)
            if os.path.exists(rt):
                fields.append(read_grd(rt)[0])
                found.append(d)
    if not fields:
        return np.empty((0, NLAT, NLON), np.float32), []
    return np.stack(fields), found


def season_days(years: Iterable[int], months: Iterable[int]) -> List[date]:
    months = set(months)
    out = []
    for y in years:
        d = date(y, 1, 1)
        while d.year == y:
            if d.month in months:
                out.append(d)
            d += timedelta(days=1)
    return out
