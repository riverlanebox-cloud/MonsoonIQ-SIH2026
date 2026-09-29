"""
Outgoing longwave radiation - the standard convection proxy for monsoon regimes.

Dataset: NOAA Interpolated OLR (Liebmann & Smith 1996, BAMS 77:1275-1277), daily, 2.5 deg,
1974-present, served by NOAA PSL THREDDS. We use the NetCDF Subset Service in grid-as-point
mode, which returns CSV for one location - no NetCDF library needed.

Anomalies are taken against the PSL 1991-2020 daily long-term mean (olr.day.ltm.1991-2020.nc).
OLR is optional: if PSL is unreachable the builder records olr_source="unavailable", fills
climatological values (anomaly 0) and the regime classifier simply learns nothing from it.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.real.openmeteo import OpenMeteoClient

logger = logging.getLogger(__name__)

OLR_URL = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/interp_OLR/olr.day.mean.nc"
LTM_URL = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/interp_OLR/olr.day.ltm.1991-2020.nc"


def _snap(lat: float, lon: float) -> Tuple[float, float]:
    """OLR is on a 2.5 deg grid; snap so nearby districts share one cached request."""
    return round(lat / 2.5) * 2.5, round(lon / 2.5) * 2.5


def _csv_to_series(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    tcol = [c for c in df.columns if c.lower().startswith("time")][0]
    vcol = [c for c in df.columns if c.lower().startswith("olr")][0]
    raw = df[tcol].astype(str).str.slice(0, 10)
    if raw.str.startswith("0001").all():
        # long-term-mean files carry year 0001, which pandas cannot represent: map the
        # month-day onto a leap reference year so day-of-year lookups still work
        raw = "2000" + raw.str.slice(4, 10)
    t = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    return pd.Series(df[vcol].astype(float).to_numpy(), index=t)


class _TextClient(OpenMeteoClient):
    """Same disk cache / retry behaviour, but for CSV bodies."""

    def get_text(self, url: str, params: dict) -> str:
        import os
        import requests
        path = self._key(url, params).replace(".json", ".csv")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        r = requests.get(url, params=params, timeout=180)
        r.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
        return r.text


def fetch_olr(points: List[Tuple[float, float]], start: date, end: date,
              cache_dir: str = "data/real/raw/olr_cache") -> Tuple[List[pd.DataFrame], str]:
    """Per point: DataFrame(date -> olr, olr_anomaly). Returns (frames, source_tag)."""
    client = _TextClient(cache_dir=cache_dir)
    snapped = sorted({_snap(*p) for p in points})
    cache: Dict[Tuple[float, float], pd.DataFrame] = {}
    try:
        for lat, lon in snapped:
            obs = _csv_to_series(client.get_text(OLR_URL, {
                "var": "olr", "latitude": lat, "longitude": lon, "accept": "csv",
                "time_start": f"{start.isoformat()}T00:00:00Z", "time_end": f"{end.isoformat()}T23:59:59Z"}))
            ltm = _csv_to_series(client.get_text(LTM_URL, {
                "var": "olr", "latitude": lat, "longitude": lon, "accept": "csv", "temporal": "all"}))
            clim = pd.Series(ltm.to_numpy(), index=ltm.index.dayofyear).groupby(level=0).mean()
            doy = np.minimum(obs.index.dayofyear, 365)
            df = pd.DataFrame({"olr": obs.to_numpy(),
                               "olr_anomaly": obs.to_numpy() - clim.reindex(doy).to_numpy()},
                              index=pd.Index(obs.index.date, name="date"))
            cache[(lat, lon)] = df
    except Exception as exc:
        logger.warning("NOAA OLR unavailable (%s) - OLR features will be climatological", exc)
        return [pd.DataFrame() for _ in points], "unavailable"
    return [cache[_snap(*p)] for p in points], "NOAA Interpolated OLR (PSL THREDDS)"
