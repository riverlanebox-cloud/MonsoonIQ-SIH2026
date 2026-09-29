"""
Table I/O that works with or without a parquet engine.

Parquet is the preferred on-disk format (pyarrow is in requirements.txt). Some locked-down
machines - a government workstation without a compiler, an air-gapped demo laptop - cannot
install pyarrow; there the archive falls back to gzip CSV next to the parquet path, and every
reader in the code base goes through read_table() so nothing else has to know.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def _csv_twin(path: str) -> str:
    root, ext = os.path.splitext(path)
    return root + ".csv.gz" if ext == ".parquet" else path


def parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def read_table(path: str) -> pd.DataFrame:
    """Read an archive table; prefers parquet, falls back to its .csv.gz twin."""
    if path.endswith((".csv", ".csv.gz")) and os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    if path.endswith(".parquet") and parquet_available() and os.path.exists(path):
        return pd.read_parquet(path)
    twin = _csv_twin(path)
    if os.path.exists(twin) and twin != path:
        return pd.read_csv(twin, low_memory=False)
    if path.endswith(".parquet") and os.path.exists(path):
        raise ImportError(
            f"{path} is parquet but no parquet engine is installed. "
            "`pip install pyarrow`, or regenerate the archive to get a .csv.gz twin.")
    raise FileNotFoundError(path)


def table_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(_csv_twin(path))


def write_table(df: pd.DataFrame, path: str) -> str:
    """Write parquet when possible, else a .csv.gz twin. Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith(".parquet") and parquet_available():
        df.to_parquet(path, index=False)
        return path
    out = _csv_twin(path)
    if out != path:
        logger.warning("No parquet engine; writing %s instead of %s", out, path)
    df.to_csv(out, index=False, compression="gzip" if out.endswith(".gz") else None)
    return out
