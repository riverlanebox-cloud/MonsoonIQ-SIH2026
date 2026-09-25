"""
Unit Tests for Verification Metrics.
Verifies RMSE, MAE, Bias, Contingency Metrics (POD, FAR, CSI, ETS, Frequency Bias),
FSS (Fractions Skill Score), Brier Score, and ROC calculations against exact analytical ground truths.
"""

import pytest
import numpy as np
from src.verification.metrics import (
    compute_continuous_metrics,
    compute_contingency_table,
    compute_dichotomous_metrics,
    compute_fractions_skill_score_2d,
    compute_reliability_diagram,
    compute_roc_curve
)


def test_continuous_metrics():
    # Exact synthetic arrays
    obs = np.array([10.0, 20.0, 30.0, 40.0])
    fcst = np.array([12.0, 18.0, 34.0, 36.0])
    # diffs = [+2, -2, +4, -4], mean diff = 0.0
    # squared diffs = [4, 4, 16, 16], mean = 10, rmse = sqrt(10) ~ 3.162
    # abs diffs = [2, 2, 4, 4], mean = 3.0
    res = compute_continuous_metrics(fcst, obs)
    assert res["bias"] == 0.0
    assert abs(res["rmse"] - 3.162) < 0.01
    assert res["mae"] == 3.0
    assert res["correlation"] > 0.95


def test_contingency_and_dichotomous_metrics():
    # Known 2x2 contingency table:
    # Hits (a) = 20, False Alarms (b) = 5, Misses (c) = 5, Correct Rejections (d) = 70
    table = {"hits": 20, "false_alarms": 5, "misses": 5, "correct_negatives": 70, "total": 100}
    m = compute_dichotomous_metrics(table)

    # POD = a / (a + c) = 20 / 25 = 0.80
    assert m["pod"] == 0.80

    # FAR = b / (a + b) = 5 / 25 = 0.20
    assert m["far"] == 0.20

    # CSI = a / (a + b + c) = 20 / 30 = 0.6667
    assert abs(m["csi"] - 0.6667) < 0.001

    # Frequency Bias = (a + b) / (a + c) = 25 / 25 = 1.00
    assert m["frequency_bias"] == 1.00

    # Hits by chance a_r = (25 * 25) / 100 = 6.25
    # ETS = (20 - 6.25) / (30 - 6.25) = 13.75 / 23.75 ~ 0.5789
    assert abs(m["ets"] - 0.5789) < 0.001


def test_fractions_skill_score():
    # Perfect agreement
    grid_obs = np.array([[100.0, 0.0], [0.0, 100.0]])
    grid_fcst = np.array([[100.0, 0.0], [0.0, 100.0]])
    fss_perfect = compute_fractions_skill_score_2d(grid_fcst, grid_obs, threshold=50.0, window_size=1)
    assert fss_perfect == 1.0

    # Spatial displacement (hit at adjacent pixel)
    # At window size 1: FSS = 0 (miss)
    # At larger window size covering both: FSS increases
    grid_fcst_displaced = np.array([[0.0, 100.0], [0.0, 0.0]])
    grid_obs_single = np.array([[100.0, 0.0], [0.0, 0.0]])

    fss_w1 = compute_fractions_skill_score_2d(grid_fcst_displaced, grid_obs_single, threshold=50.0, window_size=1)
    assert fss_w1 == 0.0

    fss_w3 = compute_fractions_skill_score_2d(grid_fcst_displaced, grid_obs_single, threshold=50.0, window_size=3)
    assert fss_w3 > 0.0 # Neighborhood scale captures proximity


def test_probabilistic_reliability_and_roc():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    obs = np.array([0, 0, 1, 1])

    rel = compute_reliability_diagram(probs, obs, n_bins=5)
    assert rel["brier_score"] < 0.05

    roc = compute_roc_curve(probs, obs)
    assert roc["auc"] == 1.0
