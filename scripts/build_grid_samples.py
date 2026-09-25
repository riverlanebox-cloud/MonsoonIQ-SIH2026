"""
Build the gridded feature archive used by the spatial (FSS) verification.

The generator keeps only a handful of 2-D fields per sampled date, and samples
dates sparsely. Fractions Skill Score needs (a) a full feature stack per grid
cell so the fitted correction models can be evaluated on the grid, and (b) enough
dates for the score to mean something. This script re-runs the generator's own
daily simulation loop with the same seed and stores land-cell feature stacks for
an explicitly chosen date sample.

Sampling policy (documented so the sample is auditable, not convenient):

    held-out years 2022-2023 : every 4th day in June-September, every 12th day otherwise
    all other years          : every 24th day
    plus                     : the named case-study dates

Storage is land-cell-flattened float32 (2281 of 3249 cells are land), which keeps
the archive small enough to commit.

Determinism is not assumed — it is checked against the grids already committed in
`data/synthetic/grid_sample_dates.npz`, and the script refuses to write if the
re-simulated meteorology differs.

Run:  PYTHONPATH=. python scripts/build_grid_samples.py
"""

import os
import sys
from datetime import datetime

import numpy as np

from src.data.synthetic_generator import SyntheticMonsoonGenerator

LEGACY = "data/synthetic/grid_sample_dates.npz"
OUT = "data/synthetic/grid_feature_samples.npz"
CHECK_FIELDS = {"true_rain": "true_rain", "nwp_d1": "raw_nwp_d1",
                "nwp_d3": "raw_nwp_d3"}

CASE_DATES = {"2018-08-15", "2016-07-26", "2017-08-10", "2019-08-08",
              "2020-07-15", "2021-06-20", "2022-03-24", "2023-07-10"}

# output name -> key in the generator's per-day record (None = from nwp_forecasts)
FIELD_KEYS = {
    "u850": "u850_field", "v850": "v850_field", "wind_speed_850": "wind_speed_850",
    "vorticity_850": "vort_field", "q500": "q500_field", "cape": "cape_field",
    "olr": "olr_field", "moisture_flux": "moisture_flux", "true_rain": "true_rain",
    "raw_nwp_d1": None, "raw_nwp_d3": None,
}

SCALAR_KEYS = {"mslp_anomaly": "mslp_anom", "trough_latitude": "trough_lat",
               "olr_anomaly": "olr_anom_mean"}

HELDOUT_YEARS = (2022, 2023)


def should_store(date: datetime, index: int) -> bool:
    key = date.strftime("%Y-%m-%d")
    if key in CASE_DATES:
        return True
    if date.year in HELDOUT_YEARS:
        if 6 <= date.month <= 9:
            return index % 4 == 0
        return index % 12 == 0
    return index % 24 == 0


def main():
    gen = SyntheticMonsoonGenerator()
    dates = gen.generate_calendar_dates()
    legacy = np.load(LEGACY, allow_pickle=True)
    legacy_dates = sorted(k for k in legacy.keys()
                          if k not in ("lats", "lons", "elevation", "land_mask"))

    land = gen.land_mask > 0.5
    n_land = int(land.sum())
    indices = np.flatnonzero(land.ravel())
    print(f"Grid {gen.land_mask.shape}, land cells {n_land}")

    records = {}
    mismatches = []
    prev_regime = 7
    for i, dt in enumerate(dates):
        day, prev_regime = gen.simulate_day(dt, prev_regime)
        if not should_store(dt, i):
            continue

        fields = {}
        for name, src in FIELD_KEYS.items():
            arr = (day["nwp_forecasts"][int(name[-1])] if src is None else day[src])
            fields[name] = np.asarray(arr, dtype=np.float32).ravel()[indices]
        for name, src in SCALAR_KEYS.items():
            fields[name] = np.full(n_land, float(day[src]), dtype=np.float32)

        payload = dict(fields)
        payload["elevation"] = gen.elevation.ravel()[indices].astype(np.float32)
        payload["slope"] = gen.slope.ravel()[indices].astype(np.float32)
        payload["dist_coast"] = gen.dist_coast.ravel()[indices].astype(np.float32)
        payload["latitude"] = gen.lat_grid.ravel()[indices].astype(np.float32)
        payload["longitude"] = gen.lon_grid.ravel()[indices].astype(np.float32)
        payload["regime_scalar"] = np.array(day["regime"], dtype=np.int16)
        records[day["date"]] = payload

        if day["date"] in legacy_dates:
            legacy_day = legacy[day["date"]].item()
            for legacy_name, new_name in CHECK_FIELDS.items():
                if legacy_name not in legacy_day or new_name not in payload:
                    continue
                a = np.asarray(legacy_day[legacy_name], dtype=np.float32).ravel()[indices]
                b = payload[new_name]
                if not np.allclose(a, b, atol=1e-4):
                    mismatches.append((day["date"], new_name, float(np.abs(a - b).max())))

        if (i + 1) % 500 == 0:
            print(f"  simulated {i + 1}/{len(dates)} days, stored {len(records)}")

    if mismatches:
        print("\nFATAL: re-simulation differs from the committed archive; not writing.")
        for key, f, d in mismatches[:10]:
            print(f"  {key} {f}: max|diff|={d}")
        sys.exit(1)

    if len(records) < 100:
        print(f"FATAL: only {len(records)} dates sampled; expected > 100.")
        sys.exit(1)

    heldout = sum(1 for k in records if int(k[:4]) in HELDOUT_YEARS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        lats=gen.lats.astype(np.float32), lons=gen.lons.astype(np.float32),
        land_index=indices.astype(np.int32),
        grid_shape=np.array(gen.land_mask.shape, dtype=np.int32),
        elevation_grid=gen.elevation.astype(np.float32),
        land_mask=gen.land_mask.astype(np.float32),
        **records,
    )
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"\nDeterminism check passed on {len(legacy_dates)} overlapping dates.")
    print(f"Stored {len(records)} date records ({heldout} in held-out years) "
          f"x {n_land} land cells x {len(FIELD_KEYS) + 8} fields")
    print(f"Wrote {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
