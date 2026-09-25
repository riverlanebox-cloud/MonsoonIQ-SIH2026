"""
MonsoonIQ Probabilistic Quantile Regressor.
Estimates P10, P50, and P90 precipitation intervals using LightGBM quantile regression.
Guarantees non-crossing quantile monotonicity: P10 <= P50 <= P90.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class QuantileRegressor:
    """Predicts probabilistic precipitation bounds (P10, P50, P90)."""

    PREDICTOR_COLS = [
        "u850", "v850", "wind_speed_850", "vorticity_850", "mslp_anomaly",
        "q500", "cape", "olr", "olr_anomaly", "moisture_flux", "trough_latitude",
        "elevation", "slope", "dist_coast", "latitude", "longitude"
    ]
    FEATURE_COLS = ["raw_nwp"] + PREDICTOR_COLS

    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90],
                 model_save_path: str = "artifacts/models/quantile_regressor.joblib"):
        self.quantiles = quantiles
        self.model_save_path = model_save_path
        self.models = {}

    def fit(self, train_df: pd.DataFrame, obs_col: str = "obs_rain_mean", nwp_col: str = "raw_nwp_d1"):
        """Fit quantile regressors for alpha=0.10, 0.50, 0.90."""
        X = train_df[self.PREDICTOR_COLS].copy()
        X["raw_nwp"] = train_df[nwp_col].to_numpy(dtype=float)
        y = train_df[obs_col].to_numpy(dtype=float)

        for q in self.quantiles:
            logger.info(f"Fitting Quantile Regressor for alpha = {q:.2f}...")
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=100,
                learning_rate=0.06,
                max_depth=5,
                num_leaves=25,
                random_state=int(42 + q * 100),
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X, y)
            self.models[q] = model

        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self, self.model_save_path)
        logger.info(f"Saved Quantile Regressor to {self.model_save_path}")
        return self

    def predict_quantiles(self, df: pd.DataFrame, nwp_col: str = "raw_nwp_d1") -> Dict[str, np.ndarray]:
        """
        Predict P10, P50, and P90 with monotonicity guarantee:
        P10 <= P50 <= P90.
        """
        X = df[self.PREDICTOR_COLS].copy()
        X["raw_nwp"] = df[nwp_col].to_numpy(dtype=float)

        preds = {}
        for q in self.quantiles:
            raw_q = np.maximum(0.0, self.models[q].predict(X))
            preds[q] = raw_q

        p10 = preds[0.10]
        p50 = preds[0.50]
        p90 = preds[0.90]

        # Enforce quantile monotonicity (rearrangement / non-crossing sorting)
        stack = np.vstack([p10, p50, p90]) # shape [3, N]
        sorted_stack = np.sort(stack, axis=0)

        return {
            "p10": sorted_stack[0, :],
            "p50": sorted_stack[1, :],
            "p90": sorted_stack[2, :]
        }

    @classmethod
    def load(cls, path: str = "artifacts/models/quantile_regressor.joblib") -> "QuantileRegressor":
        """Load fitted quantile regressor."""
        return joblib.load(path)
