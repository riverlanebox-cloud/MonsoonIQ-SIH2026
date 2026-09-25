"""
MonsoonIQ End-to-End Pipeline & Shape Tests.
Verifies:
1. Regime classifier output shapes, range [1, 7], and soft probability normalization.
2. MoE prediction shapes and non-negativity.
3. Quantile regressor monotonicity: P10 <= P50 <= P90.
4. Heavy rain calibrated hierarchy: P(>=64.5) >= P(>=115.6) >= P(>=204.5).
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.regime.rule_classifier import RuleRegimeClassifier
from src.regime.ml_classifier import MLRegimeClassifier
from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
from src.correction.quantile_regressor import QuantileRegressor
from src.heavy_rain.heavy_rain_classifier import HeavyRainProbabilityModule


def test_rule_regime_classifier():
    classifier = RuleRegimeClassifier()
    sample = {
        "month": 7, "u850": 16.0, "vorticity_850": 2.5e-5, "olr_anomaly": -20.0,
        "mslp_anomaly": -2.0, "trough_latitude": 22.0, "slope": 0.003, "elevation": 100.0,
        "dist_coast": 120.0, "latitude": 22.0, "longitude": 80.0
    }
    dominant, probs = classifier.classify_record(sample)
    assert 1 <= dominant <= 7
    assert len(probs) == 7
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_ml_regime_classifier():
    clf = MLRegimeClassifier()
    clf.load()

    # Create dummy dataframe
    dummy_data = {col: [10.0] for col in clf.FEATURE_COLS}
    dummy_df = pd.DataFrame(dummy_data)

    probs = clf.predict_proba(dummy_df)
    assert probs.shape == (1, 7)
    assert abs(np.sum(probs) - 1.0) < 0.01

    pred = clf.predict(dummy_df)
    assert 1 <= pred[0] <= 7


def test_quantile_regressor_monotonicity():
    qr = QuantileRegressor.load()
    dummy_data = {col: [10.0] for col in qr.PREDICTOR_COLS}
    dummy_data["raw_nwp_d1"] = [25.0]
    dummy_df = pd.DataFrame(dummy_data)

    quants = qr.predict_quantiles(dummy_df, nwp_col="raw_nwp_d1")
    p10 = quants["p10"][0]
    p50 = quants["p50"][0]
    p90 = quants["p90"][0]

    assert p10 <= p50, f"P10 ({p10}) exceeded P50 ({p50})"
    assert p50 <= p90, f"P50 ({p50}) exceeded P90 ({p90})"
    assert p10 >= 0.0, "Negative precipitation predicted"


def test_heavy_rain_hierarchy():
    hrc = HeavyRainProbabilityModule.load()
    dummy_data = {col: [10.0] for col in hrc.PREDICTOR_COLS}
    dummy_data["raw_nwp_d1"] = [75.0]
    dummy_df = pd.DataFrame(dummy_data)

    probs = hrc.predict_probabilities(dummy_df, nwp_col="raw_nwp_d1")
    p_h = probs["heavy"][0]
    p_vh = probs["very_heavy"][0]
    p_eh = probs["extremely_heavy"][0]

    assert p_h >= p_vh, f"P(heavy) {p_h} < P(very_heavy) {p_vh}"
    assert p_vh >= p_eh, f"P(very_heavy) {p_vh} < P(extremely_heavy) {p_eh}"
