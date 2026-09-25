"""
MonsoonIQ Heavy Rainfall Classifier Module.
Trains separate calibrated binary classifiers for IMD severe rainfall thresholds:
- >= 64.5 mm/day (Heavy Rainfall)
- >= 115.6 mm/day (Very Heavy Rainfall)
- >= 204.5 mm/day (Extremely Heavy Rainfall)
Applies scale_pos_weight for severe class imbalance and isotonic calibration
to yield reliable, well-calibrated probabilities.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


class HeavyRainProbabilityModule:
    """Calibrated probability models for IMD extreme rainfall thresholds."""

    THRESHOLDS = {
        "heavy": 64.5,
        "very_heavy": 115.6,
        "extremely_heavy": 204.5
    }

    PREDICTOR_COLS = [
        "u850", "v850", "wind_speed_850", "vorticity_850", "mslp_anomaly",
        "q500", "cape", "olr", "olr_anomaly", "moisture_flux", "trough_latitude",
        "elevation", "slope", "dist_coast", "latitude", "longitude"
    ]
    FEATURE_COLS = ["raw_nwp"] + PREDICTOR_COLS

    def __init__(self, model_save_path: str = "artifacts/models/heavy_rain_module.joblib"):
        self.model_save_path = model_save_path
        self.classifiers = {}
        self.calibrators = {}

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            target_col: str = "obs_rain_max", nwp_col: str = "raw_nwp_d1") -> Dict[str, Any]:
        """Fit calibrated classifiers for each IMD threshold."""
        X_train = train_df[self.PREDICTOR_COLS].copy()
        X_train["raw_nwp"] = train_df[nwp_col].to_numpy(dtype=float)
        y_max_train = train_df[target_col].to_numpy(dtype=float)

        X_val = val_df[self.PREDICTOR_COLS].copy()
        X_val["raw_nwp"] = val_df[nwp_col].to_numpy(dtype=float)
        y_max_val = val_df[target_col].to_numpy(dtype=float)

        metrics = {}

        for key, threshold in self.THRESHOLDS.items():
            y_bin_train = (y_max_train >= threshold).astype(int)
            y_bin_val = (y_max_val >= threshold).astype(int)

            pos_count = np.sum(y_bin_train)
            neg_count = len(y_bin_train) - pos_count
            scale_pos = max(1.0, float(neg_count / max(1, pos_count)))

            logger.info(f"Training threshold {key} (>={threshold} mm): {pos_count} positives ({pos_count/len(y_bin_train)*100:.2f}%), scale_pos_weight={scale_pos:.2f}")

            clf = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=25,
                scale_pos_weight=min(scale_pos, 25.0), # prevent over-inflation
                random_state=42 + int(threshold),
                n_jobs=-1,
                verbose=-1
            )

            clf.fit(
                X_train, y_bin_train,
                eval_set=[(X_val, y_bin_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
            self.classifiers[key] = clf

            # Isotonic probability calibration on validation partition
            if np.sum(y_bin_val) >= 5:
                try:
                    cal = CalibratedClassifierCV(estimator=clf, method="isotonic", cv="prefit")
                    cal.fit(X_val, y_bin_val)
                    self.calibrators[key] = cal
                except Exception as e:
                    logger.warning(f"Isotonic calibration failed for {key}: {e}")
                    self.calibrators[key] = clf
            else:
                self.calibrators[key] = clf

            # Evaluate Brier Score & AUC
            p_val = self.calibrators[key].predict_proba(X_val)[:, 1]
            b_score = brier_score_loss(y_bin_val, p_val)
            try:
                auc = roc_auc_score(y_bin_val, p_val) if np.sum(y_bin_val) > 0 else 0.5
            except Exception:
                auc = 0.5

            metrics[key] = {
                "threshold_mm": threshold,
                "train_positives": int(pos_count),
                "val_brier_score": round(float(b_score), 4),
                "val_auc": round(float(auc), 4)
            }

        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self, self.model_save_path)
        logger.info(f"Saved calibrated Heavy Rain Module to {self.model_save_path}")

        return metrics

    def predict_probabilities(self, df: pd.DataFrame, nwp_col: str = "raw_nwp_d1") -> Dict[str, np.ndarray]:
        """Predict calibrated probabilities of exceeding 64.5, 115.6, and 204.5 mm/day."""
        X = df[self.PREDICTOR_COLS].copy()
        X["raw_nwp"] = df[nwp_col].to_numpy(dtype=float)

        probs = {}
        for key in self.THRESHOLDS:
            p = self.calibrators[key].predict_proba(X)[:, 1]
            probs[key] = np.clip(p, 0.0, 1.0)

        # Enforce physical hierarchy: P(>=64.5) >= P(>=115.6) >= P(>=204.5)
        probs["very_heavy"] = np.minimum(probs["very_heavy"], probs["heavy"])
        probs["extremely_heavy"] = np.minimum(probs["extremely_heavy"], probs["very_heavy"])

        return probs

    @classmethod
    def load(cls, path: str = "artifacts/models/heavy_rain_module.joblib") -> "HeavyRainProbabilityModule":
        """Load fitted Heavy Rain module."""
        return joblib.load(path)
