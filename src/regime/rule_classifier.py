"""
MonsoonIQ Rule-Based Weather Regime Classifier.
Applies expert meteorological rules from configs/regime_rules.yaml
based on IMD synoptic climatology.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class RuleRegimeClassifier:
    """Classifies weather regimes based on physical atmospheric and topographic thresholds."""

    REGIME_NAMES = {
        1: "Active Monsoon",
        2: "Break Monsoon",
        3: "Monsoon Low/Depression",
        4: "Orographic",
        5: "Coastal",
        6: "Western Disturbance",
        7: "Weak/Normal"
    }

    def __init__(self, config_path: str = "configs/regime_rules.yaml"):
        self.config_path = config_path
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("regimes", {})

    def classify_record(self, row: Dict[str, Any]) -> Tuple[int, Dict[int, float]]:
        """
        Evaluate physical rule conditions for a single spatio-temporal observation.
        Returns:
            dominant_regime (int): 1 through 7
            soft_scores (dict): pseudo-probability / rule score for each regime
        """
        month = int(row.get("month", 7))
        u850 = float(row.get("u850", 10.0))
        vorticity = float(row.get("vorticity_850", 1.0e-5))
        olr_anom = float(row.get("olr_anomaly", 0.0))
        mslp_anom = float(row.get("mslp_anomaly", 0.0))
        trough_lat = float(row.get("trough_latitude", 22.0))
        slope = float(row.get("slope", 0.005))
        elev = float(row.get("elevation", 100.0))
        dist_coast = float(row.get("dist_coast", 100.0))
        lat = float(row.get("latitude", 20.0))
        lon = float(row.get("longitude", 78.0))

        scores = {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.15}

        is_monsoon_month = (6 <= month <= 9)
        is_transition_or_winter = (month in [1, 2, 3, 4, 5, 10, 11, 12])

        # 3. Monsoon Low / Depression
        # Characterized by intense cyclonic vorticity (>4.0e-5 s^-1) and negative MSLP anomaly (< -3.5 hPa)
        if (6 <= month <= 10) and (vorticity >= 3.8e-5 or mslp_anom <= -3.2):
            dep_score = 0.5
            if vorticity >= 4.5e-5:
                dep_score += 0.3
            if mslp_anom <= -4.0:
                dep_score += 0.2
            scores[3] = max(scores[3], dep_score)

        # 6. Western Disturbance
        # North India (lat >= 25, lon <= 82), upper-level cyclonic anomaly, non-monsoon/transition months
        if is_transition_or_winter and lat >= 24.5 and lon <= 82.5:
            wd_score = 0.4
            if vort_flag := (vorticity >= 2.0e-5):
                wd_score += 0.3
            if mslp_anom <= -1.5 or olr_anom <= -15.0:
                wd_score += 0.2
            scores[6] = max(scores[6], wd_score)

        # 4. Orographic
        # Western Ghats ridge (elev > 300m, slope > 0.010, west coast lon 73-77) or Himalayan slopes
        if (slope >= 0.008 or elev >= 400.0) and u850 >= 10.0:
            oro_score = 0.35
            if slope >= 0.012:
                oro_score += 0.35
            if lon <= 76.5 and 8.0 <= lat <= 20.5:
                oro_score += 0.25
            scores[4] = max(scores[4], oro_score)

        # 5. Coastal Convergence
        # Within 65 km of coastline
        if dist_coast <= 65.0 and not (slope >= 0.012 and elev >= 500.0):
            coast_score = 0.45
            if dist_coast <= 35.0:
                coast_score += 0.3
            scores[5] = max(scores[5], coast_score)

        # 2. Break Monsoon
        # Central India dry, trough moved north to foothills (trough_lat >= 27.0), positive OLR anomaly
        if is_monsoon_month and trough_lat >= 26.5 and olr_anom >= 8.0:
            brk_score = 0.5
            if trough_lat >= 28.0:
                brk_score += 0.25
            if u850 <= 8.5:
                brk_score += 0.2
            scores[2] = max(scores[2], brk_score)

        # 1. Active Monsoon
        # Vigorous low level jet (u850 >= 13), central trough (18-26N), negative OLR anomaly
        if is_monsoon_month and u850 >= 12.0 and 18.0 <= trough_lat <= 26.0 and olr_anom <= -10.0:
            act_score = 0.5
            if u850 >= 15.0:
                act_score += 0.25
            if vorticity >= 2.0e-5:
                act_score += 0.2
            scores[1] = max(scores[1], act_score)

        # Soft softmax-like normalization of scores
        total_score = sum(scores.values())
        soft_probs = {k: round(v / total_score, 4) for k, v in scores.items()}

        # Dominant regime is the one with highest rule score
        dominant = max(scores.keys(), key=lambda k: scores[k])
        return dominant, soft_probs

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each row of a DataFrame and append rule_regime and probabilities."""
        dominant_list = []
        prob_cols = {f"p_regime_{k}": [] for k in range(1, 8)}

        for _, row in df.iterrows():
            dom, probs = self.classify_record(row.to_dict())
            dominant_list.append(dom)
            for k in range(1, 8):
                prob_cols[f"p_regime_{k}"].append(probs[k])

        out_df = df.copy()
        out_df["rule_regime"] = dominant_list
        out_df["rule_regime_name"] = [self.REGIME_NAMES[k] for k in dominant_list]
        for col_name, p_vals in prob_cols.items():
            out_df[col_name] = p_vals
        return out_df


if __name__ == "__main__":
    clf = RuleRegimeClassifier()
    test_obs = {
        "month": 7, "u850": 17.5, "vorticity_850": 3.2e-5, "olr_anomaly": -22.0,
        "mslp_anomaly": -2.0, "trough_latitude": 22.5, "slope": 0.003, "elevation": 120.0,
        "dist_coast": 150.0, "latitude": 21.5, "longitude": 82.0
    }
    dom, probs = clf.classify_record(test_obs)
    print("Dominant Regime:", clf.REGIME_NAMES[dom])
    print("Regime Soft Probabilities:", probs)
