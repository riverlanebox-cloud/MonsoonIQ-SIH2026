"""
Physically defined weather-regime labels for the real archive.

On synthetic data the regime is a latent variable of the simulator. On real data there is no
oracle, so labels are *derived* from observations and analysis with published criteria, then
the ML classifier learns to predict them from issue-time information. The labels never look at
the NWP forecast, so the regime is not defined by the error it is supposed to explain.

Regime (priority order, per district-day):
  6 Western Disturbance  Oct-May, lat >= 25N, lon <= 82E, 500 hPa height anomaly <= -35 gpm
  3 Monsoon Low/Depression  MSLP anomaly <= -3 hPa and 850 hPa relative vorticity >= 3e-5 s-1
                            (IMD: a low is a closed isobar with 1-2 hPa depth; a depression has
                            winds 17-27 kt - we use the pressure/vorticity signature)
  4 Orographic   terrain >= 350 m and upslope 850 hPa flow >= 6 m/s (westerly component on the
                 Western Ghats, southerly component on the Himalaya and NE hills)
  5 Coastal      within 65 km of the coast and onshore 850 hPa flow >= 5 m/s
  1 Active / 2 Break   core-monsoon-zone state from Rajeevan, Bhate & Jaswal (2010): standardised
                 daily rainfall anomaly over the CMZ >= +1 SD (active) or <= -1 SD (break) for at
                 least 3 consecutive days, June-September
  7 Weak/Normal  everything else

Thresholds live in configs/data_sources.yaml (regime_labels) and in DEFAULTS below.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Approximate core monsoon zone of Rajeevan et al. (2010, their Fig. 1): lon, lat vertices.
CMZ_POLYGON = [(69.0, 21.0), (71.5, 23.5), (76.0, 25.5), (80.0, 27.0), (83.0, 27.0), (87.5, 24.5),
               (88.0, 22.0), (85.0, 20.0), (82.0, 18.5), (78.0, 18.0), (74.0, 18.5), (71.0, 19.5),
               (69.0, 21.0)]

DEFAULTS = {
    "cmz_threshold_sd": 1.0, "cmz_min_run_days": 3,
    "depression_mslp_anom_max_hpa": -3.0, "depression_vorticity_min": 3.0e-5,
    "orographic_elevation_min_m": 350.0, "orographic_upslope_wind_min": 6.0,
    "coastal_distance_max_km": 65.0, "coastal_onshore_wind_min": 5.0,
    "wd_months": [1, 2, 3, 4, 5, 10, 11, 12], "wd_z500_anom_max_gpm": -35.0,
}

REGIME_NAMES = {1: "Active Monsoon", 2: "Break Monsoon", 3: "Monsoon Low/Depression",
                4: "Orographic", 5: "Coastal", 6: "Western Disturbance", 7: "Weak/Normal"}


def cmz_mask(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    from src.data.geometry import contains
    gx, gy = np.meshgrid(lons, lats)
    geom = {"type": "Polygon", "coordinates": [CMZ_POLYGON]}
    return contains(geom, gx.ravel(), gy.ravel()).reshape(gy.shape)


def cmz_rainfall(fields: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Area-mean rainfall over the CMZ for each day of a (days, lat, lon) stack."""
    m = cmz_mask(lats, lons)
    w = np.cos(np.radians(lats))[:, None] * m
    vals = np.where(np.isnan(fields), 0.0, fields) * w
    valid = (~np.isnan(fields)) * w
    return vals.sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1e-9)


def standardised_cmz_index(cmz: pd.Series, clim_years: Optional[List[int]] = None,
                           window: int = 31) -> pd.Series:
    """
    (R - mean_doy) / sd_doy with a 31-day smoothed day-of-year climatology, computed from
    `clim_years` only (the training years) so held-out seasons do not inform their own labels.
    """
    s = cmz.copy()
    s.index = pd.to_datetime(s.index)
    base = s if not clim_years else s[s.index.year.isin(clim_years)]
    doy = base.index.dayofyear
    mean = base.groupby(doy).mean().reindex(range(1, 367))
    sd = base.groupby(doy).std().reindex(range(1, 367))
    pad = lambda x: pd.concat([x.iloc[-window:], x, x.iloc[:window]])  # noqa: E731
    mean_s = pad(mean.interpolate(limit_direction="both")).rolling(window, center=True, min_periods=1).mean().iloc[window:-window]
    sd_s = pad(sd.interpolate(limit_direction="both")).rolling(window, center=True, min_periods=1).mean().iloc[window:-window]
    d = s.index.dayofyear
    z = (s.to_numpy() - mean_s.reindex(d).to_numpy()) / np.maximum(sd_s.reindex(d).to_numpy(), 1e-6)
    return pd.Series(z, index=cmz.index, name="cmz_index")


def spells(index: pd.Series, threshold: float = 1.0, min_run: int = 3) -> pd.Series:
    """+1 active, -1 break, 0 otherwise; a spell needs `min_run` consecutive qualifying days."""
    idx = pd.to_datetime(index.index)
    state = np.where(index.to_numpy() >= threshold, 1, np.where(index.to_numpy() <= -threshold, -1, 0))
    out = np.zeros_like(state)
    i = 0
    n = len(state)
    while i < n:
        j = i
        while j + 1 < n and state[j + 1] == state[i] and (idx[j + 1] - idx[j]).days == 1:
            j += 1
        if state[i] != 0 and (j - i + 1) >= min_run:
            out[i:j + 1] = state[i]
        i = j + 1
    return pd.Series(out, index=index.index, name="cmz_spell")


def label(df: pd.DataFrame, cfg: Optional[Dict] = None) -> pd.DataFrame:
    """
    Add `regime` (1-7), `regime_name`, and `regime_rule` (the criterion that fired).

    Needs columns: month, latitude, longitude, elevation, dist_coast, cmz_spell,
    lab_u850, lab_v850, lab_vort850, lab_mslp_anomaly, lab_z500_anomaly.
    """
    c = {**DEFAULTS, **(cfg or {})}
    n = len(df)
    regime = np.full(n, 7, dtype=int)
    rule = np.array(["no criterion met"] * n, dtype=object)
    month = df["month"].to_numpy()
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    u = df["lab_u850"].to_numpy(float)
    v = df["lab_v850"].to_numpy(float)
    jjas = np.isin(month, [6, 7, 8, 9])

    spell = df["cmz_spell"].to_numpy()
    m = jjas & (spell == 1)
    regime[m], rule[m] = 1, "CMZ index >= +1 SD for >= 3 days (Rajeevan 2010)"
    m = jjas & (spell == -1)
    regime[m], rule[m] = 2, "CMZ index <= -1 SD for >= 3 days (Rajeevan 2010)"

    west = lon < 77.5
    onshore = np.where(west, u, -u)                                  # Arabian Sea vs Bay coasts
    onshore = np.where((lon > 84) & (lat > 19), np.maximum(onshore, v), onshore)   # Odisha/Bengal
    m = (df["dist_coast"].to_numpy(float) <= c["coastal_distance_max_km"]) & (onshore >= c["coastal_onshore_wind_min"])
    regime[m], rule[m] = 5, "coast <= 65 km and onshore 850 hPa flow"

    upslope = np.where(lat >= 25.0, v, u)                              # Himalaya/NE vs Ghats
    upslope = np.where((lon > 89) & (lat > 23), np.maximum(u, v), upslope)
    m = (df["elevation"].to_numpy(float) >= c["orographic_elevation_min_m"]) & (upslope >= c["orographic_upslope_wind_min"])
    regime[m], rule[m] = 4, "terrain >= 350 m and upslope 850 hPa flow"

    m = (df["lab_mslp_anomaly"].to_numpy(float) <= c["depression_mslp_anom_max_hpa"]) & \
        (df["lab_vort850"].to_numpy(float) >= c["depression_vorticity_min"])
    regime[m], rule[m] = 3, "MSLP anomaly <= -3 hPa and 850 hPa vorticity >= 3e-5"

    m = np.isin(month, c["wd_months"]) & (lat >= 25) & (lon <= 82) & \
        (df["lab_z500_anomaly"].to_numpy(float) <= c["wd_z500_anom_max_gpm"])
    regime[m], rule[m] = 6, "Oct-May, NW India, 500 hPa trough (z500 anomaly <= -35 gpm)"

    out = df.copy()
    out["regime"] = regime
    out["regime_name"] = [REGIME_NAMES[r] for r in regime]
    out["regime_rule"] = rule
    return out
