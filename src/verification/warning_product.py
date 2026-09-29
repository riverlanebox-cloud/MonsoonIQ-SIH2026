"""
Verification of the product a district actually receives: the IMD-coloured warning.

The continuous correction targets the district-MEAN rainfall, while IMD heavy-rain warnings
are about the heaviest rain anywhere in the district (the district MAXIMUM of the 0.25 deg
cells). Scoring a mean forecast against a maximum observation makes every mean-rainfall
system look blind to heavy rain. The warning product combines what the system is built to
use for that question - the calibrated exceedance probabilities, the P90 of the predictive
distribution and the corrected amount - through the same rule the console and bulletins use
(AdvisoryGenerator.determine_alert_level). This module scores that product against the
observed district maximum, per lead, with day-block bootstrap intervals, alongside the raw
model read the same way (raw district rainfall >= threshold).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.api.advisory import AdvisoryGenerator
from src.verification.stratified import day_blocks, paired_delta, score_with_interval

CATEGORY_MM = {"green": 0.0, "yellow": 64.5, "orange": 115.6, "red": 204.5}
THRESHOLDS = (64.5, 115.6, 204.5)


def warning_categories(corrected, p90, p_heavy, p_very_heavy) -> np.ndarray:
    return np.array([AdvisoryGenerator.determine_alert_level(
        mean_rain=float(corrected[i]), max_rain=float(max(p90[i], corrected[i])),
        p_heavy=float(p_heavy[i]), p_very_heavy=float(p_very_heavy[i])) for i in range(len(corrected))])


def verify_warning_product(test_df: pd.DataFrame, lead_bundle, posteriors: Optional[np.ndarray],
                           iters: int = 400) -> Dict[str, Any]:
    if lead_bundle is None:
        return {"available": False, "reason": "per-lead bundle not fitted"}
    y_max = test_df["obs_rain_max"].to_numpy(float)
    groups = day_blocks(test_df["date"].to_numpy())
    out: Dict[str, Any] = {
        "available": True,
        "observed_quantity": "district maximum of IMD 0.25 deg cells (obs_rain_max)",
        "product_rule": "AdvisoryGenerator.determine_alert_level (same rule as console / bulletin / CAP)",
        "event_forecast": {"64.5": "category >= yellow", "115.6": "category >= orange", "204.5": "red"},
        "leads": {},
    }
    for lead in lead_bundle.available_leads():
        preds = lead_bundle.predict_all_systems(test_df, lead, regime_probs=posteriors)
        probs = lead_bundle.predict_probabilities(test_df, lead)
        quants = lead_bundle.predict_quantiles(test_df, lead)
        cats = warning_categories(preds["monsooniq"], quants["p90"], probs["heavy"], probs["very_heavy"])
        product_mm = np.array([CATEGORY_MM[c] for c in cats])
        raw = test_df[f"raw_nwp_d{lead}"].to_numpy(float)
        entry: Dict[str, Any] = {"category_counts": {c: int((cats == c).sum()) for c in CATEGORY_MM}}
        for thr in THRESHOLDS:
            key = f"{thr}"
            p = score_with_interval(product_mm, y_max, thr, groups, metric="csi", iters=iters)
            r = score_with_interval(raw, y_max, thr, groups, metric="csi", iters=iters)
            entry[key] = {
                "warning_product": {k: p[k] for k in ("pod", "far", "csi", "ets", "frequency_bias",
                                                      "event_count", "reliability", "csi_ci95")},
                "raw_nwp": {k: r[k] for k in ("pod", "far", "csi", "ets", "frequency_bias", "csi_ci95")},
                "delta_vs_raw": paired_delta(product_mm, raw, y_max, thr, groups, metric="csi", iters=iters),
            }
        out["leads"][f"day_{lead}"] = entry
    return out
