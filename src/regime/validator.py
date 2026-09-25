"""
MonsoonIQ Regime Validation & Cross-Check Tool.
Cross-checks rule-labeled active/break spells and depression periods
against IMD monsoon bulletins and known depression tracks CSV.
Reports agreement scores, precision, recall, and contingency metrics.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class RegimeSpellValidator:
    """Validates classified regime sequences against external ground-truth bulletin records."""

    def __init__(self, depression_tracks_path: str = "data/sample_depression_tracks.csv"):
        self.tracks_path = depression_tracks_path
        self.depression_tracks = self._load_tracks()

    def _load_tracks(self) -> pd.DataFrame:
        if not os.path.exists(self.tracks_path):
            logger.warning(f"Depression tracks CSV not found at {self.tracks_path}")
            return pd.DataFrame()
        df = pd.read_csv(self.tracks_path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df

    def evaluate_depression_agreement(self, regime_df: pd.DataFrame,
                                     date_col: str = "date",
                                     regime_col: str = "regime",
                                     depression_class_id: int = 3) -> Dict[str, Any]:
        """
        Cross-check classified Monsoon Low/Depression days against known IMD tracks.
        """
        if self.depression_tracks.empty:
            return {"error": "No external depression records loaded"}

        known_dates = set(self.depression_tracks["date"].unique())
        daily_regimes = regime_df.groupby(date_col)[regime_col].agg(lambda x: (x == depression_class_id).mean()).to_dict()

        total_known = len(known_dates)
        hits = 0
        misses = 0

        for d in known_dates:
            if d in daily_regimes:
                # If depression regime was active in >= 15% of districts or dominant
                if daily_regimes[d] >= 0.15:
                    hits += 1
                else:
                    misses += 1

        # False alarms: days classified as depression with no known track
        false_alarms = 0
        correct_negatives = 0
        for d, frac in daily_regimes.items():
            if d not in known_dates:
                if frac >= 0.25:
                    false_alarms += 1
                else:
                    correct_negatives += 1

        pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

        report = {
            "total_known_depression_dates": total_known,
            "matched_depression_hits": hits,
            "missed_depression_dates": misses,
            "unverified_depression_days": false_alarms,
            "pod_hit_rate": round(pod, 4),
            "false_alarm_ratio": round(far, 4),
            "critical_success_index": round(csi, 4),
            "percentage_agreement": round(pod * 100.0, 2),
            "known_tracks_matched": self.depression_tracks["system_name"].tolist()
        }
        return report

    def evaluate_spell_durations(self, regime_df: pd.DataFrame, regime_id: int = 1) -> Dict[str, Any]:
        """
        Evaluate climatological spell length statistics (e.g. Active spell length 3-7 days).
        """
        dates_sorted = sorted(regime_df["date"].unique())
        daily_active = [
            (regime_df[regime_df["date"] == d]["regime"] == regime_id).mean() >= 0.25
            for d in dates_sorted
        ]

        spell_lengths = []
        current_len = 0
        for act in daily_active:
            if act:
                current_len += 1
            else:
                if current_len > 0:
                    spell_lengths.append(current_len)
                    current_len = 0
        if current_len > 0:
            spell_lengths.append(current_len)

        if not spell_lengths:
            return {"mean_spell_days": 0, "max_spell_days": 0, "total_spells": 0}

        return {
            "mean_spell_days": round(float(np.mean(spell_lengths)), 2),
            "median_spell_days": float(np.median(spell_lengths)),
            "max_spell_days": int(np.max(spell_lengths)),
            "total_spells": len(spell_lengths)
        }


if __name__ == "__main__":
    validator = RegimeSpellValidator()
    print("Loaded depression tracks:", len(validator.depression_tracks))
    if not validator.depression_tracks.empty:
        print(validator.depression_tracks.head(3))
