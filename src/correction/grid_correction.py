"""
Grid-native regime-aware correction.

The district product and the grid product are not the same object. District
corrections are fitted on district-average rainfall; pushing those adjustments
onto grid cells through a district map produces a mosaic of steps and measurably
*hurts* spatial skill (see the `district_transfer` rows in the verification
summary — the negative result is reported, not hidden).

This module fits the correction where it is issued:

    features  : the 16 atmospheric/terrain predictors at the cell
                + raw NWP rainfall at the cell
                + the 7 regime posteriors for that day
    target    : gridded observed rainfall at the cell
    training  : all sampled grid dates in the training/validation years
    verified  : sampled grid dates in the held-out years

Regime is a day-level property in this archive, so applying the regime classifier
per cell is consistent; the correction itself is cell-level, which is what FSS
needs to score.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

logger = logging.getLogger(__name__)

GRID_NPZ = "data/synthetic/grid_feature_samples.npz"
DEFAULT_PATH = "artifacts/models/grid_correction.joblib"

PREDICTORS = [
    "u850", "v850", "wind_speed_850", "vorticity_850", "mslp_anomaly", "q500",
    "cape", "olr", "olr_anomaly", "moisture_flux", "trough_latitude",
    "elevation", "slope", "dist_coast", "latitude", "longitude",
]
META_KEYS = {"lats", "lons", "land_index", "grid_shape", "elevation_grid", "land_mask"}


def _records(npz) -> List[str]:
    return sorted(k for k in npz.keys() if k not in META_KEYS)


def _frame(npz, key: str) -> pd.DataFrame:
    rec = npz[key].item()
    data = {name: np.asarray(rec[name], dtype=float) for name in PREDICTORS}
    data["raw_nwp"] = np.asarray(rec["raw_nwp_d1"], dtype=float)
    return pd.DataFrame(data)


class GridCorrectionModel:
    """LightGBM correction fitted on grid cells, with regime posteriors as inputs."""

    REGIME_NAMES = ["active", "break", "depression", "orographic", "coastal",
                    "western_disturbance", "weak_normal"]

    def __init__(self, save_path: str = DEFAULT_PATH, npz_path: str = GRID_NPZ,
                 train_years: Tuple[int, ...] = (2016, 2017, 2018, 2019, 2020, 2021),
                 test_years: Tuple[int, ...] = (2022, 2023)):
        self.save_path = save_path
        self.npz_path = npz_path
        self.train_years = train_years
        self.test_years = test_years
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names: List[str] = []
        self.training_meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _with_regime(df: pd.DataFrame, clf) -> pd.DataFrame:
        posteriors = clf.predict_proba(df)
        out = df.copy()
        for i, name in enumerate(GridCorrectionModel.REGIME_NAMES):
            out[f"p_regime_{name}"] = posteriors[:, i]
        return out

    def _load(self) -> Tuple[Any, List[str]]:
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"grid archive not found: {self.npz_path}")
        npz = np.load(self.npz_path, allow_pickle=True)
        return npz, _records(npz)

    # -------------------------------------------------------------------- fit
    def fit(self, clf, max_rows: int = 400_000) -> "GridCorrectionModel":
        npz, keys = self._load()
        frames, targets, dates = [], [], []
        for key in keys:
            year = int(key[:4])
            if year not in self.train_years:
                continue
            df = self._with_regime(_frame(npz, key), clf)
            rec = npz[key].item()
            frames.append(df)
            targets.append(np.asarray(rec["true_rain"], dtype=float))
            dates.append(key)

        if not frames:
            raise RuntimeError("no training grid dates found")

        X = pd.concat(frames, ignore_index=True)
        y = np.concatenate(targets)
        if len(X) > max_rows:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_rows, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y[idx]

        self.feature_names = list(X.columns)
        logger.info("Fitting grid correction on %d cell-days from %d dates",
                    len(X), len(dates))
        self.model = lgb.LGBMRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=7, num_leaves=63,
            subsample=0.85, colsample_bytree=0.85, min_child_samples=40,
            random_state=42, n_jobs=-1, verbose=-1)
        self.model.fit(X, y)
        self.training_meta = {
            "cell_days": int(len(X)),
            "dates": int(len(dates)),
            "train_years": list(self.train_years),
            "features": self.feature_names,
            "train_rmse": round(float(np.sqrt(np.mean((self.model.predict(X) - y) ** 2))), 4),
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        joblib.dump(self, self.save_path)
        logger.info("Saved grid correction to %s (train RMSE %.3f)",
                    self.save_path, self.training_meta["train_rmse"])
        return self

    # ---------------------------------------------------------------- predict
    def predict_field(self, npz, key: str, clf, raw: Optional[np.ndarray] = None) -> np.ndarray:
        df = self._with_regime(_frame(npz, key), clf)
        corrected = np.maximum(0.0, self.model.predict(df[self.feature_names]))
        return corrected

    def predict_isocorrected(self, df: pd.DataFrame, clf) -> np.ndarray:
        """Inverse: apply a district-scale model to grid cells (used for comparisons)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "GridCorrectionModel":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return joblib.load(path)

    def verification_fields(self, clf, years: Tuple[int, ...] = (2022, 2023)
                            ) -> Dict[str, Dict[str, np.ndarray]]:
        """{date: {'truth':…, 'raw_nwp':…, 'grid_correction':…}} on land cells."""
        npz, keys = self._load()
        out = {}
        for key in keys:
            if int(key[:4]) not in years:
                continue
            rec = npz[key].item()
            out[key] = {
                "truth": np.asarray(rec["true_rain"], dtype=float),
                "raw_nwp": np.asarray(rec["raw_nwp_d1"], dtype=float),
                "grid_correction": self.predict_field(npz, key, clf),
            }
        return out
