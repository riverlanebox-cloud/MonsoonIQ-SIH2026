"""
Tests for the day-block bootstrap grouping and the stratified scoring helper.

These guard the statistical plumbing: if `day_blocks` silently emits an empty
block, every confidence interval in the product is computed over a distribution
that includes an empty sample, and the reported calendar-day count is wrong.
"""

import numpy as np
import pandas as pd

from src.verification.stratified import day_blocks, stratified_categorical


def _mm(pred, mask):
    """Rainfall amounts in mm, which is what the categorical scorer consumes.

    `compute_contingency_table` thresholds the *prediction* at the rainfall
    threshold, so passing 0/1 binary flags silently produces a zero contingency
    table. Predictions must be in the same units as the observations.
    """
    return np.where(mask, pred, 0.0)


def _frame(y, regimes, preds, start="2022-06-01"):
    dates = pd.date_range(start, periods=len(y) // 53, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": np.repeat(dates, 53)[:len(y)],
        "obs_rain_max": y,
        "obs_rain_mean": y * 0.7,
        "regime": regimes,
        **{f"pred_{k}": v for k, v in preds.items()},
    })


def test_day_blocks_returns_one_block_per_unique_day():
    days = ["2022-06-01", "2022-06-02", "2022-06-01", "2022-06-02", "2022-06-02"]
    blocks = day_blocks(days)
    assert len(blocks) == 2, "one block per unique day"
    assert all(len(b) > 0 for b in blocks), "no block may be empty"
    assert sorted(np.concatenate(blocks).tolist()) == [0, 1, 2, 3, 4]


def test_day_blocks_handles_edge_shapes():
    assert len(day_blocks(["b", "a", "c"])) == 3
    assert len(day_blocks(["a"])) == 1
    assert len(day_blocks([])) == 0


def test_day_blocks_covers_a_two_year_archive_shape():
    dates = pd.date_range("2022-06-01", periods=522, freq="D").strftime("%Y-%m-%d")
    days = np.repeat(dates, 53)
    blocks = day_blocks(days)
    assert len(blocks) == 522
    assert sum(len(b) for b in blocks) == len(days)


def test_stratified_suppresses_thin_strata_and_reports_thick_ones():
    rng = np.random.RandomState(0)
    n = 4293
    regimes = np.ones(n, dtype=int)
    thin_from = 3500
    regimes[thin_from:] = 2  # ~793 rows, thin on events
    # rainfall in mm: ~5% of rows are heavy events
    y = np.where(rng.rand(n) < 0.05, 120.0 + rng.rand(n) * 60, rng.rand(n) * 8)
    df = _frame(y, regimes, {
        "raw_nwp": np.where(rng.rand(n) < 0.60, 90.0 + rng.rand(n) * 40, rng.rand(n) * 10),
        "monsooniq": np.where(y >= 64.5, 95.0 + rng.rand(n) * 50, rng.rand(n) * 5),
    })
    out = stratified_categorical(
        df, systems={"raw_nwp": df["pred_raw_nwp"].to_numpy(),
                     "monsooniq": df["pred_monsooniq"].to_numpy()},
        threshold=64.5, stratify_col="regime",
        system_order=["raw_nwp", "monsooniq"], paired=("monsooniq", "raw_nwp"),
        interval_iters=60,
    )
    assert set(out) == {"1", "2"}
    assert out["1"]["reportable"] is True
    assert out["1"]["improvement"]["delta"] > 0.4, "the better system must show a positive delta"
    assert out["1"]["systems"]["monsooniq"]["csi"] > out["1"]["systems"]["raw_nwp"]["csi"]
    assert out["1"]["day_count"] > 0
    if not out["2"]["reportable"]:
        assert "systems" in out["2"], "thin strata stay in the payload but are not advertised"


def test_stratified_thin_stratum_is_flagged_not_scored():
    rng = np.random.RandomState(2)
    n = 53 * 20
    y = np.zeros(n)
    y[:2] = 200.0  # 2 events only - below the reporting floor
    df = _frame(y, np.ones(n, dtype=int),
                {"raw_nwp": np.zeros(n), "monsooniq": np.full(n, 120.0)})

    out = stratified_categorical(
        df, systems={"raw_nwp": np.zeros(n), "monsooniq": np.full(n, 120.0)},
        threshold=64.5, stratify_col="regime", system_order=["raw_nwp", "monsooniq"],
        paired=("monsooniq", "raw_nwp"), interval_iters=30,
    )
    entry = out["1"]
    assert entry["reportable"] is False
    assert entry["reliability"] == "insufficient"
    assert entry["improvement"].get("suppressed") is True
