"""
MonsoonIQ Mixture-of-Experts (MoE) Bias-Correction Engine.
Implements:
1. Seven regime experts, each = base forecast + regime residual model. The base is
   either regime quantile mapping ("qm", the original design, used on the synthetic
   archive) or the global gradient-boosting model's out-of-fold prediction ("gbm",
   used on the real archive). On real IMD/GFS data, 2021-2023, quantile mapping at
   district-day scale is worse than the raw model (it inflates variance where the
   forecast-observation correlation is weak), so regime experts built on it cannot
   recover; on the GBM base each regime learns only the part of the error that is
   regime-specific, shrunk towards zero when the regime has few training days.
   The choice was made on the 2023 validation season, not the test seasons.
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
from src.compat import lgb
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

    SHRINK_ROWS = 2000  # rows of a regime at which its residual gets half weight

    def __init__(self, model_save_path: str = "artifacts/models/mixture_of_experts.joblib",
                 expert_base: Optional[str] = None):
        self.model_save_path = model_save_path
        if expert_base is None:
            from src import config
            expert_base = "gbm" if config.MODE == "real" else "qm"
        self.expert_base = expert_base
        self.expert_weight = {r: 1.0 for r in self.EXPERT_REGIMES}

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

        if self.expert_base == "gbm":
            return self._fit_gbm_experts(train_df, val_df, nwp_train, obs_train, regimes_train,
                                         nwp_val, obs_val, regimes_val, X_train_global, X_val_global)

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
        base_mode = getattr(self, "expert_base", "qm")
        for idx, (r_id, expert) in enumerate(self.experts.items()):
            if base_mode == "gbm":
                X_expert = X_global.copy()
                X_expert["qm_base"] = global_lgb_pred
                res = expert.predict_residual(X_expert) * self.expert_weight.get(r_id, 1.0)
                expert_preds[:, idx] = np.maximum(0.0, global_lgb_pred + res)
                continue
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

    def _fit_gbm_experts(self, train_df, val_df, nwp_train, obs_train, regimes_train,
                         nwp_val, obs_val, regimes_val, X_train_global, X_val_global):
        """Regime residual experts on the global GBM's out-of-fold prediction."""
        days = train_df["date"].astype(str).to_numpy()
        uniq = np.unique(days)
        fold_of_day = {d: i % 4 for i, d in enumerate(uniq)}  # contiguous-ish day blocks
        fold = np.array([fold_of_day[d] for d in days])
        oof = np.zeros(len(train_df))
        params = self.global_lgb.get_params()
        for k in range(4):
            tr, te = fold != k, fold == k
            m = lgb.LGBMRegressor(**params)
            m.fit(X_train_global[tr], obs_train[tr])
            oof[te] = np.maximum(0.0, m.predict(X_train_global[te]))
        base_val = np.maximum(0.0, self.global_lgb.predict(X_val_global))

        expert_train_scores = {}
        for r_id, expert in self.experts.items():
            mt, mv = regimes_train == r_id, regimes_val == r_id
            X_r = X_train_global[mt].copy()
            X_r["qm_base"] = oof[mt]
            X_rv, res_v = None, None
            if mv.sum() > 10:
                X_rv = X_val_global[mv].copy()
                X_rv["qm_base"] = base_val[mv]
                res_v = obs_val[mv] - base_val[mv]
            expert.fit(X_r, obs_train[mt] - oof[mt], X_rv, res_v)
            n = int(mt.sum())
            self.expert_weight[r_id] = n / (n + self.SHRINK_ROWS)
            expert_train_scores[expert.regime_name] = n
            logger.info(f"Expert {r_id} ({expert.regime_name}) on GBM base: {n} rows, "
                        f"shrinkage weight {self.expert_weight[r_id]:.2f}")

        self.is_fitted = True
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(self, self.model_save_path)
        logger.info(f"Saved complete MoE system to {self.model_save_path}")
        return {"expert_samples": expert_train_scores, "expert_base": "gbm",
                "expert_weights": {self.EXPERT_REGIMES[k]: round(v, 3) for k, v in self.expert_weight.items()}}

    @classmethod
    def load(cls, path: str = "artifacts/models/mixture_of_experts.joblib") -> "MonsoonIQMixtureOfExperts":
        """Load trained MoE instance."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"MoE artifact not found at {path}")
        return joblib.load(path)
