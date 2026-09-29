"""
Central data-mode configuration.

MonsoonIQ runs on one of two archives:

    real       IMD 0.25 deg gridded rainfall (observations) + NOAA GFS 0.5/1 deg
               forecasts, June-September 2021-2025. Built by
               `scripts/build_real_archive.py` from the bundles fetched by
               `scripts/fetch/` (see docs/REAL_DATA.md).
    synthetic  the seeded physically-parameterised simulator in src/data/.

Selection: the MONSOONIQ_DATA environment variable ("real" / "synthetic").
When unset, the real archive is used if it has been built, otherwise synthetic.
Every path and every train/validation/test split in the pipeline reads from here,
so switching mode never needs a code change.
"""

import os
from typing import Dict, List

_REAL_DIR = "data/real"
_SYN_DIR = "data/synthetic"

SPLITS: Dict[str, Dict[str, List[int]]] = {
    # 5 monsoon seasons: fit on 2021-2022, tune on 2023, report on 2024-2025.
    "real": {"train": [2021, 2022], "val": [2023], "test": [2024, 2025]},
    "synthetic": {"train": [2016, 2017, 2018, 2019, 2020], "val": [2021],
                  "test": [2022, 2023]},
}


def _archive_exists(d: str) -> bool:
    return any(os.path.exists(os.path.join(d, f"district_daily.{ext}"))
               for ext in ("parquet", "csv.gz"))


def data_mode() -> str:
    env = os.environ.get("MONSOONIQ_DATA", "").strip().lower()
    if env in ("real", "synthetic"):
        return env
    return "real" if _archive_exists(_REAL_DIR) else "synthetic"


MODE = data_mode()
DATA_DIR = _REAL_DIR if MODE == "real" else _SYN_DIR
ARCHIVE = os.path.join(DATA_DIR, "district_daily.parquet")
GRID_NPZ = os.path.join(DATA_DIR, "grid_feature_samples.npz")
METADATA = os.path.join(DATA_DIR, "dataset_metadata.json")

TRAIN_YEARS = SPLITS[MODE]["train"]
VAL_YEARS = SPLITS[MODE]["val"]
TEST_YEARS = SPLITS[MODE]["test"]


def period_label() -> str:
    """Human label for the held-out period, e.g. 'real held-out period 2024-2025'."""
    lo, hi = min(TEST_YEARS), max(TEST_YEARS)
    return f"{MODE} held-out period {lo}–{hi}"
