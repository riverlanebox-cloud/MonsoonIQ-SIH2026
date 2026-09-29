"""
Run profile: which archive the whole chain (train -> evaluate -> console -> API) uses.

MonsoonIQ ships two archives that share one schema (see docs/DATASETS.md):

  synthetic  data/synthetic/district_daily.parquet   committed, reproducible benchmark
  real       data/real/district_daily.parquet        built on your machine from IMD 0.25 deg
             (or .csv.gz)                            gridded rainfall + NWP forecasts
                                                     (scripts/fetch_real_data.py)

Select with the environment variable MONSOONIQ_PROFILE=synthetic|real (default synthetic).
Every path in the code base that points at the archive or at artifacts/ is routed
through P(), so the two profiles never overwrite each other's models or metrics:
the real profile writes to artifacts/real/... and reads data/real/... .
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

PROFILE = os.getenv("MONSOONIQ_PROFILE", "synthetic").strip().lower()
if PROFILE not in ("synthetic", "real"):
    raise ValueError(f"MONSOONIQ_PROFILE must be 'synthetic' or 'real', got {PROFILE!r}")

REAL_DIR = os.getenv("MONSOONIQ_REAL_DIR", "data/real")

# Geometry: real Census-2011 polygons (DataMeet, CC-BY 2.5 IN) for display and for the
# real archive; the synthetic generator keeps the legacy study polygons it was
# generated with so the committed benchmark stays bit-reproducible.
STUDY_GEOJSON = "data/geojson/india_districts.geojson"
ALL_DISTRICTS_GEOJSON = "data/geojson/india_districts_census2011.geojson"
SYNTHETIC_GEOJSON = "data/geojson/synthetic_study_districts.geojson"


def _real_archive() -> str:
    """The most recently built real archive (parquet, or its CSV twin on machines without pyarrow)."""
    found = [os.path.join(REAL_DIR, n) for n in ("district_daily.parquet", "district_daily.csv.gz",
                                                 "district_daily.csv")
             if os.path.exists(os.path.join(REAL_DIR, n))]
    if found:
        return max(found, key=os.path.getmtime)
    return os.path.join(REAL_DIR, "district_daily.parquet")


def P(path: str) -> str:
    """Map a canonical (synthetic-profile) path to the active profile's path."""
    if PROFILE == "synthetic":
        return path
    norm = path.replace("\\", "/")
    if norm == "data/synthetic/district_daily.parquet":
        return _real_archive()
    if norm.startswith("data/synthetic/"):
        return os.path.join(REAL_DIR, norm[len("data/synthetic/"):])
    if norm.startswith("artifacts/"):
        return os.path.join("artifacts", "real", norm[len("artifacts/"):])
    return path


DATA_PATH = P("data/synthetic/district_daily.parquet")

# Time-based split. Synthetic: fixed. Real: from the archive metadata written by
# the builder (it knows which seasons were actually fetched), else configs/data_sources.yaml.
_SYNTHETIC_SPLIT = {"train": [2016, 2017, 2018, 2019, 2020], "val": [2021], "test": [2022, 2023]}


def _real_split() -> Dict[str, List[int]]:
    meta_path = os.path.join(REAL_DIR, "dataset_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("split"):
            return {k: [int(y) for y in v] for k, v in meta["split"].items()}
    try:
        import yaml
        with open("configs/data_sources.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {k: [int(y) for y in v] for k, v in cfg["split"].items()}
    except Exception:
        return {"train": [2021, 2022, 2023], "val": [2024], "test": [2025]}


SPLIT = _SYNTHETIC_SPLIT if PROFILE == "synthetic" else _real_split()
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = SPLIT["train"], SPLIT["val"], SPLIT["test"]


def provenance_label() -> str:
    return "SYNTHETIC_PHYSICALLY_PLAUSIBLE" if PROFILE == "synthetic" else "REAL_OBSERVED_IMD_NWP"
