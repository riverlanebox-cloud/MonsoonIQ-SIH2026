"""
MonsoonIQ Mixture-of-Experts (MoE) Bias-Correction Engine.
Implements:
1. Seven regime experts (Quantile Mapping baseline + LightGBM residual learning).
2. Soft blending: Forecast = sum_{k=1}^7 P(R_k) * Expert_k.
3. Baseline 1: Raw NWP
4. Baseline 2: Global Quantile Mapping (regime-agnostic)
5. Baseline 3: Global LightGBM (regime-agnostic)
6. Model persistence and batch inference.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Optional, Tuple

from src.correction.quantile_mapping import EmpiricalQuantileMapper, RegimeQuantileMapper
from src.correction.residual_expert import RegimeResidualExpert

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class MonsoonIQMixtureOfExperts:
    """Regime-Aware Mixture-of-Experts Post-Processing System."""

    EXPERT_REGIMES = {
        1: "Active Monsoon",
        2: "Break Monsoon",
        3: "Monsoon Low/Depression",
        4: "Orographic",
        5: "Coastal",
        6: "Western Disturbance",
        7: "Weak/Normal"
    }

    PREDICTOR_COLS = [
        "u850", "v850", "wind_speed_850", "vorticity_850", "mslp_anomaly",
        "q500", "cape", "olr", "olr_anomaly", "moisture_flux", "trough_latitude",
        "elevation", "slope", "dist_coast", "latitude", "longitude"
    ]
    FEATURE_COLS = ["raw_nwp"] + PREDICTOR_COLS

    def __init__(self, model_save_path: str = "artifacts/models/mixture_of_experts.joblib"):
        self.model_save_path = model_save_path

        # MoE components
        self.regime_qm = RegimeQuantileMapper()
        self.experts = {
            r: RegimeResidualExpert(r, name)
            for r, name in self.EXPERT_REGIMES.items()
        }

        # Baseline 2: Global Quantile Mapping (no regime conditioning)
        self.global_qm = EmpiricalQuantileMapper()

        # Baseline 3: Global LightGBM (no regime conditioning)
        self.global_lgb = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            nwp_col: str = "raw_nwp_d1", obs_col: str = "obs_rain_mean",
            regime_col: str = "regime") -> Dict[str, Any]:
        """
        Train MoE experts and global baselines strictly on training data.
        """
        logger.info(f"Training MonsoonIQ MoE and Baselines on {len(train_df)} records...")

        nwp_train = train_df[nwp_col].to_numpy(dtype=float)
        obs_train = train_df[obs_col].to_numpy(dtype=float)
        regimes_train = train_df[regime_col].to_numpy(dtype=int)

        nwp_val = val_df[nwp_col].to_numpy(dtype=float)
        obs_val = val_df[obs_col].to_numpy(dtype=float)
        regimes_val = val_df[regime_col].to_numpy(dtype=int)

        # 1. Fit Global QM Baseline
        self.global_qm.fit(nwp_train, obs_train)
        logger.info("Global Quantile Mapping baseline fitted.")

        # 2. Fit Global LightGBM Baseline
        X_train_global = train_df[self.PREDICTOR_COLS].copy()
        X_train_global["raw_nwp"] = nwp_train
        X_val_global = val_df[self.PREDICTOR_COLS].copy()
        X_val_global["raw_nwp"] = nwp_val

        self.global_lgb.fit(
            X_train_global, obs_train,
            eval_set=[(X_val_global, obs_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        logger.info("Global LightGBM baseline fitted.")

        # 3. Fit Regime Quantile Mappers
        self.regime_qm.fit(nwp_train, obs_train, regimes_train)

        # 4. Compute regime QM outputs and residuals for each expert
        expert_train_scores = {}
        for r_id, expert in self.experts.items():
            r_mask_train = (regimes_train == r_id)
            r_mask_val = (regimes_val == r_id)

            # In regime r, what was the QM output?
            qm_train_r = self.regime_qm.transform_single_regime(nwp_train[r_mask_train], r_id)
            residual_train_r = obs_train[r_mask_train] - qm_train_r

            X_r_train = train_df.loc[r_mask_train, self.PREDICTOR_COLS].copy()
            X_r_train["raw_nwp"] = nwp_train[r_mask_train]
            X_r_train["qm_base"] = qm_train_r

            X_r_val = None
            residual_val_r = None
            if np.sum(r_mask_val) > 10:
                qm_val_r = self.regime_qm.transform_single_regime(nwp_val[r_mask_val], r_id)
                residual_val_r = obs_val[r_mask_val] - qm_val_r
                X_r_val = val_df.loc[r_mask_val, self.PREDICTOR_COLS].copy()
                X_r_val["raw_nwp"] = nwp_val[r_mask_val]
                X_r_val["qm_base"] = qm_val_r

            expert.fit(X_r_train, residual_train_r, X_r_val, residual_val_r)
            expert_train_scores[expert.regime_name] = len(X_r_train)
            logger.info(f"Expert {r_id} ({expert.regime_name}) trained on {len(X_r_train)} samples.")

        self.is_fitted = True

        # Save trained MoE system
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self, self.model_save_path)
        logger.info(f"Saved complete MoE system to {self.model_save_path}")

        return {"status": "success", "expert_samples": expert_train_scores}

    def predict_all_systems(self, df: pd.DataFrame, nwp_col: str = "raw_nwp_d1",
                            regime_probs: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Produce predictions from all 4 systems for strict comparison:
        - raw_nwp
        - global_qm
        - global_lgb
        - monsooniq (Regime-aware soft-blended MoE)
        """
        nwp_vals = df[nwp_col].to_numpy(dtype=float)
        N = len(df)

        # Baseline 1: Raw NWP
        raw_pred = np.maximum(0.0, nwp_vals)

        # Baseline 2: Global QM
        global_qm_pred = self.global_qm.transform(nwp_vals)

        # Baseline 3: Global LightGBM
        X_global = df[self.PREDICTOR_COLS].copy()
        X_global["raw_nwp"] = nwp_vals
        global_lgb_pred = np.maximum(0.0, self.global_lgb.predict(X_global))

        # MonsoonIQ: Soft-Blended MoE
        if regime_probs is None:
            # Fallback uniform or rule probs
            regime_probs = np.ones((N, 7)) / 7.0

        expert_preds = np.zeros((N, 7), dtype=float)
        for idx, (r_id, expert) in enumerate(self.experts.items()):
            qm_r = self.regime_qm.transform_single_regime(nwp_vals, r_id)
            X_expert = df[self.PREDICTOR_COLS].copy()
            X_expert["raw_nwp"] = nwp_vals
            X_expert["qm_base"] = qm_r
            residual_hat = expert.predict_residual(X_expert)
            expert_preds[:, idx] = np.maximum(0.0, qm_r + residual_hat)

        # Soft blending across 7 regimes: sum_k P(R_k) * Expert_k
        monsooniq_pred = np.sum(regime_probs * expert_preds, axis=1)
        monsooniq_pred = np.maximum(0.0, monsooniq_pred)

        return {
            "raw_nwp": raw_pred,
            "global_qm": global_qm_pred,
            "global_lgb": global_lgb_pred,
            "monsooniq": monsooniq_pred,
            "expert_breakdown": expert_preds
        }

    @classmethod
    def load(cls, path: str = "artifacts/models/mixture_of_experts.joblib") -> "MonsoonIQMixtureOfExperts":
        """Load trained MoE instance."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"MoE artifact not found at {path}")
        return joblib.load(path)
