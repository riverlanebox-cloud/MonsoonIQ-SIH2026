"""
Optional-dependency shims.

The pipeline is written against LightGBM, pyarrow, shapely and shap (see
requirements.txt), and that is what `make setup` installs. Some build
environments cannot install them (no package index, no compiler toolchain). In
those environments these shims keep the pipeline runnable with standard
scientific-Python equivalents, and every training manifest records which backend
produced it (`GBM_BACKEND`), so a number is never silently attributed to the
wrong library.

    lgb            lightgbm, or a LightGBM-compatible wrapper over scikit-learn's
                   HistGradientBoosting{Regressor,Classifier}
    read_table     parquet via pandas/pyarrow, else the committed .csv.gz sibling
    write_table    both formats where possible
    points_in_polygon  shapely-free point-in-polygon (matplotlib.path)
"""

import os
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- GBMs
# The scikit-learn-backed classes are always defined (not only when LightGBM is
# missing) so that models trained with them unpickle anywhere.
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def _hgb_kwargs(kw: dict, n_default: int = 100) -> dict:
    depth = kw.get("max_depth", -1)
    return {
        "max_iter": int(kw.get("n_estimators", n_default)),
        "learning_rate": float(kw.get("learning_rate", 0.1)),
        "max_depth": depth if depth and depth > 0 else None,
        "max_leaf_nodes": int(kw.get("num_leaves", 31)),
        "min_samples_leaf": int(kw.get("min_child_samples", 20)),
        "l2_regularization": float(kw.get("reg_lambda", 0.0)),
        "random_state": kw.get("random_state", None),
    }


class _EarlyStop:
    def __init__(self, stopping_rounds: int = 15, verbose: bool = False):
        self.stopping_rounds = stopping_rounds


class _Base:
    _is_shim = True

    def _prep_es(self, kwargs: dict, est) -> None:
        cbs = kwargs.pop("callbacks", None) or []
        es = [c for c in cbs if isinstance(c, _EarlyStop) or type(c).__name__ == "_EarlyStop"]
        if es and kwargs.get("eval_set") is not None:
            est.set_params(early_stopping=True, n_iter_no_change=es[0].stopping_rounds,
                           validation_fraction=0.15)
        else:
            est.set_params(early_stopping=False)
        for k in ("eval_set", "eval_metric", "eval_names", "verbose"):
            kwargs.pop(k, None)

    def predict(self, X):
        return self._est.predict(X)

    def get_params(self, deep: bool = True):
        return dict(self._kw)

    @property
    def n_features_in_(self):
        return self._est.n_features_in_


class SkLGBMRegressor(_Base):
    """LightGBM-compatible regressor over HistGradientBoostingRegressor."""

    def __init__(self, **kw: Any):
        self._kw = kw
        params = _hgb_kwargs(kw)
        if kw.get("objective", "regression") == "quantile":
            params.update(loss="quantile", quantile=float(kw.get("alpha", 0.5)))
        self._est = HistGradientBoostingRegressor(**params)

    def fit(self, X, y, **kwargs):
        self._prep_es(kwargs, self._est)
        self._est.fit(X, y, sample_weight=kwargs.get("sample_weight"))
        return self


class SkLGBMClassifier(_Base):
    """LightGBM-compatible classifier over HistGradientBoostingClassifier."""

    def __init__(self, **kw: Any):
        self._kw = kw
        params = _hgb_kwargs(kw)
        spw = kw.get("scale_pos_weight")
        self._spw = float(spw) if spw else None
        if kw.get("class_weight") is not None:
            params["class_weight"] = kw["class_weight"]
        self._est = HistGradientBoostingClassifier(**params)

    def fit(self, X, y, **kwargs):
        self._prep_es(kwargs, self._est)
        sw = kwargs.get("sample_weight")
        if self._spw and sw is None:
            sw = np.where(np.asarray(y) == 1, self._spw, 1.0)
        self._est.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, X):
        return self._est.predict_proba(X)

    @property
    def classes_(self):
        return self._est.classes_


# names used by models pickled before the rename
LGBMRegressor, LGBMClassifier = SkLGBMRegressor, SkLGBMClassifier


class _LgbNamespace:
    LGBMRegressor = SkLGBMRegressor
    LGBMClassifier = SkLGBMClassifier

    @staticmethod
    def early_stopping(stopping_rounds: int = 15, verbose: bool = False):
        return _EarlyStop(stopping_rounds, verbose)


_FORCE_SK = os.environ.get("MONSOONIQ_GBM", "").lower() == "sklearn"
try:  # pragma: no cover - depends on environment
    if _FORCE_SK:
        raise ImportError("MONSOONIQ_GBM=sklearn")
    import lightgbm as lgb  # type: ignore
    GBM_BACKEND = "lightgbm"
except Exception:  # pragma: no cover
    lgb = _LgbNamespace()  # type: ignore
    GBM_BACKEND = "sklearn-hist-gradient-boosting"
    logger.info("Using scikit-learn HistGradientBoosting backend (LightGBM not used).")

# -------------------------------------------------------------------- tables
try:  # pragma: no cover
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except Exception:  # pragma: no cover
    HAS_PARQUET = False


def _csv_sibling(path: str) -> str:
    base = path[:-len(".parquet")] if path.endswith(".parquet") else path
    return base + ".csv.gz"


def read_table(path: str) -> pd.DataFrame:
    """Read an archive written by `write_table`, preferring parquet."""
    if HAS_PARQUET and os.path.exists(path):
        return pd.read_parquet(path)
    alt = _csv_sibling(path)
    if os.path.exists(alt):
        return pd.read_csv(alt, compression="gzip")
    if os.path.exists(path):
        raise RuntimeError(f"{path} needs pyarrow to read; install requirements.txt")
    raise FileNotFoundError(path)


def write_table(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(_csv_sibling(path), index=False, compression="gzip", float_format="%.6g")
    if HAS_PARQUET:
        df.to_parquet(path, index=False)
    return _csv_sibling(path)


def table_exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(_csv_sibling(path))


def table_path_for_hash(path: str) -> str:
    return path if (HAS_PARQUET and os.path.exists(path)) else _csv_sibling(path)


# -------------------------------------------------------------- geometry
def points_in_polygon(rings, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for one polygon given as a list of (N,2) rings."""
    from matplotlib.path import Path
    pts = np.column_stack([lons.ravel(), lats.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for ring in rings:
        inside ^= Path(np.asarray(ring)).contains_points(pts)
    return inside.reshape(np.shape(lons))


# ------------------------------------------------------------------- shap
try:  # pragma: no cover
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # explanations degrade to model feature importances
