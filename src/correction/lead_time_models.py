"""
Per-lead-time correction bundle (Day 1 … Day 5).

The honest version of a multi-day product: a separate regime-aware correction is
fitted for each forecast lead, instead of pushing Day 2-5 values through a model
trained on Day 1. Each lead gets its own

    * regime-conditional quantile mapping + per-regime residual experts
    * regime-agnostic baselines (global quantile mapping, global LightGBM)
    * calibrated exceedance-probability models for 64.5 / 115.6 / 204.5 mm
    * quantile regressors for the P10 / P50 / P90 band

so the dashboard's Day 1-5 control is backed by five separately verified models
rather than one model applied five times.

Fitting all five leads takes ~1 minute on the synthetic archive; the fitted
bundle is a single joblib file.
"""

import os
import shutil
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional

from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
from src.correction.quantile_regressor import QuantileRegressor
from src.heavy_rain.heavy_rain_classifier import HeavyRainProbabilityModule

logger = logging.getLogger(__name__)

DEFAULT_LEADS = (1, 2, 3, 4, 5)
CACHE_DIR = "artifacts/models/_lead_cache"


class LeadTimeCorrectionBundle:
    """Five independently fitted regime-aware correction systems, one per lead."""

    def __init__(self, leads: tuple = DEFAULT_LEADS,
                 save_path: str = "artifacts/models/lead_time_bundle.joblib",
                 cache_dir: str = CACHE_DIR):
        self.leads = tuple(leads)
        self.save_path = save_path
        self.cache_dir = cache_dir
        self.systems: Dict[int, Dict[str, Any]] = {}
        self.training_summary: Dict[str, Any] = {}

    @staticmethod
    def _nwp_col(lead: int) -> str:
        return f"raw_nwp_d{lead}"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
            obs_col: str = "obs_rain_mean", obs_max_col: str = "obs_rain_max",
            regime_col: str = "regime") -> "LeadTimeCorrectionBundle":
        os.makedirs(self.cache_dir, exist_ok=True)
        for lead in self.leads:
            nwp_col = self._nwp_col(lead)
            if nwp_col not in train_df.columns:
                logger.warning("Column %s absent; skipping lead %d", nwp_col, lead)
                continue

            logger.info("=== Fitting lead Day %d (%s) ===", lead, nwp_col)
            moe = MonsoonIQMixtureOfExperts(
                model_save_path=os.path.join(self.cache_dir, f"moe_d{lead}.joblib"))
            info = moe.fit(train_df, val_df, nwp_col=nwp_col, obs_col=obs_col,
                           regime_col=regime_col)

            qr = QuantileRegressor(
                model_save_path=os.path.join(self.cache_dir, f"qr_d{lead}.joblib"))
            qr.fit(train_df, obs_col=obs_col, nwp_col=nwp_col)

            hrc = HeavyRainProbabilityModule(
                model_save_path=os.path.join(self.cache_dir, f"hrc_d{lead}.joblib"))
            hrc_metrics = hrc.fit(train_df, val_df, target_col=obs_max_col, nwp_col=nwp_col)

            self.systems[lead] = {"moe": moe, "quantile": qr, "probability": hrc,
                                  "nwp_col": nwp_col}
            self.training_summary[f"day_{lead}"] = {
                "nwp_column": nwp_col,
                "expert_training_samples": info.get("expert_samples", {}),
                "probability_models": hrc_metrics,
            }

        self.save()
        # The per-lead cache exists only because the component classes persist
        # themselves on fit(); the bundle is the single artifact that matters.
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        return self

    def save(self) -> str:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        joblib.dump(self, self.save_path)
        logger.info("Saved lead-time bundle (%d leads) to %s", len(self.systems), self.save_path)
        return self.save_path

    @classmethod
    def load(cls, path: str = "artifacts/models/lead_time_bundle.joblib") -> "LeadTimeCorrectionBundle":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Lead-time bundle not found at {path}")
        return joblib.load(path)

    def available_leads(self) -> List[int]:
        return sorted(self.systems.keys())

    def predict_all_systems(self, df: pd.DataFrame, lead: int,
                            regime_probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if lead not in self.systems:
            raise KeyError(f"Lead {lead} not fitted; available={self.available_leads()}")
        sysd = self.systems[lead]
        return sysd["moe"].predict_all_systems(df, nwp_col=sysd["nwp_col"],
                                               regime_probs=regime_probs)

    def predict_probabilities(self, df: pd.DataFrame, lead: int) -> Dict[str, np.ndarray]:
        sysd = self.systems[lead]
        return sysd["probability"].predict_probabilities(df, nwp_col=sysd["nwp_col"])

    def predict_quantiles(self, df: pd.DataFrame, lead: int) -> Dict[str, np.ndarray]:
        sysd = self.systems[lead]
        return sysd["quantile"].predict_quantiles(df, nwp_col=sysd["nwp_col"])

    def verification_matrix(self, df: pd.DataFrame, regime_probs: Optional[np.ndarray] = None,
                            thresholds=(64.5, 115.6)) -> Dict[str, Any]:
        """
        Per-lead skill of every system. This is the table behind the dashboard's
        "skill vs lead time" exhibit.
        """
        from src.verification.metrics import (
            compute_continuous_metrics, compute_contingency_table,
            compute_dichotomous_metrics)

        y = df["obs_rain_mean"].to_numpy(float)
        y_max = df["obs_rain_max"].to_numpy(float)
        out: Dict[str, Any] = {}
        for lead in self.available_leads():
            preds = self.predict_all_systems(df, lead, regime_probs=regime_probs)
            entry: Dict[str, Any] = {"lead_days": lead, "systems": {}}
            for name in ("raw_nwp", "global_qm", "global_lgb", "monsooniq"):
                cont = compute_continuous_metrics(preds[name], y)
                sys_entry = {"rmse": cont["rmse"], "mae": cont["mae"], "bias": cont["bias"]}
                for thr in thresholds:
                    d = compute_dichotomous_metrics(
                        compute_contingency_table(preds[name], y_max, thr))
                    sys_entry[f"csi_{thr}"] = d["csi"]
                    sys_entry[f"ets_{thr}"] = d["ets"]
                    sys_entry[f"pod_{thr}"] = d["pod"]
                    sys_entry[f"far_{thr}"] = d["far"]
                entry["systems"][name] = sys_entry
            out[f"day_{lead}"] = entry
        return out
