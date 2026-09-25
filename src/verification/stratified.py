"""
Stratified verification with sample-size gating and block-bootstrap intervals.

Two problems this module exists to solve:

1. `compute_bootstrap_ci` in `metrics.py` resamples individual district-days
   i.i.d. Adjacent districts on one day are the same weather system and adjacent
   days are autocorrelated, so that interval is far too narrow. Everything here
   resamples whole calendar days ("blocks").
2. Categorical scores on rare events are meaningless on small samples. A regime
   with 0 heavy-rainfall events in the verification period would print
   CSI = 0.000 and look like a scored result. Every table cell here carries its
   own event count and an explicit reliability class, and the evaluator is
   expected to hide or grey out `insufficient` cells rather than report them.

Reliability classes
    reliable      >= 30 events
    indicative    >= MIN_EVENTS (10) and < 30
    insufficient  < MIN_EVENTS  -- not reportable
"""

import numpy as np
from typing import Dict, Any, List, Optional, Sequence

from src.verification.metrics import (
    compute_continuous_metrics,
    compute_contingency_table,
    compute_dichotomous_metrics,
)

MIN_EVENTS = 10
RELIABLE_EVENTS = 30
DEFAULT_ITERS = 400
DEFAULT_SEED = 20260922

CATEGORICAL_KEYS = ("pod", "far", "csi", "ets", "frequency_bias")


def reliability_class(event_count: int) -> str:
    if event_count >= RELIABLE_EVENTS:
        return "reliable"
    if event_count >= MIN_EVENTS:
        return "indicative"
    return "insufficient"


def day_blocks(days: Sequence) -> List[np.ndarray]:
    """Group row indices by calendar day so bootstrap resamples whole days.

    The leading boundary must be dropped: `np.split` on a boundary list that
    starts at 0 emits an empty first block, which would both inflate the
    calendar-day count and let the bootstrap draw an empty day.
    """
    days = np.asarray(days)
    if days.size == 0:
        return []
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_days[1:] != sorted_days[:-1]])
    if boundaries.size and boundaries[0] == 0:
        boundaries = boundaries[1:]
    return np.split(order, boundaries)


def _resample(groups: List[np.ndarray], rng: np.random.RandomState) -> np.ndarray:
    picks = rng.randint(0, len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in picks])


def score(pred: np.ndarray, obs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Categorical score card with its own sample bookkeeping."""
    table = compute_contingency_table(pred, obs, threshold)
    metrics = compute_dichotomous_metrics(table)
    metrics["event_count"] = int(table["hits"] + table["misses"])
    metrics["sample_count"] = int(table["total"])
    metrics["reliability"] = reliability_class(metrics["event_count"])
    metrics["reportable"] = metrics["reliability"] != "insufficient"
    return metrics


def score_with_interval(pred: np.ndarray, obs: np.ndarray, threshold: float,
                        groups: List[np.ndarray], metric: str = "csi",
                        iters: int = DEFAULT_ITERS,
                        seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Categorical score plus a day-block bootstrap interval on that score."""
    point = score(pred, obs, threshold)
    rng = np.random.RandomState(seed)
    samples = np.empty(iters)
    for i in range(iters):
        idx = _resample(groups, rng)
        samples[i] = compute_dichotomous_metrics(
            compute_contingency_table(pred[idx], obs[idx], threshold))[metric]
    out = dict(point)
    out[f"{metric}_ci95"] = [round(float(np.percentile(samples, 2.5)), 4),
                             round(float(np.percentile(samples, 97.5)), 4)]
    out[f"{metric}_spread"] = round(float(samples.std()), 4)
    return out


def paired_delta(pred_a: np.ndarray, pred_b: np.ndarray, obs: np.ndarray,
                 threshold: float, groups: List[np.ndarray], metric: str = "csi",
                 iters: int = DEFAULT_ITERS, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Paired improvement of A over B with a day-block bootstrap interval."""
    rng = np.random.RandomState(seed)
    deltas = np.empty(iters)
    for i in range(iters):
        idx = _resample(groups, rng)
        a = compute_dichotomous_metrics(compute_contingency_table(pred_a[idx], obs[idx], threshold))[metric]
        b = compute_dichotomous_metrics(compute_contingency_table(pred_b[idx], obs[idx], threshold))[metric]
        deltas[i] = a - b

    table_a = compute_contingency_table(pred_a, obs, threshold)
    table_b = compute_contingency_table(pred_b, obs, threshold)
    delta_point = (compute_dichotomous_metrics(table_a)[metric]
                   - compute_dichotomous_metrics(table_b)[metric])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": metric,
        "delta": round(float(delta_point), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_positive": round(float((deltas > 0).mean()), 3),
        "significant_95": bool(lo > 0.0 or hi < 0.0),
        "favours": "A" if delta_point > 0 else ("B" if delta_point < 0 else "tie"),
        "a_events": int(table_a["hits"] + table_a["misses"]),
        "b_events": int(table_b["hits"] + table_b["misses"]),
    }


def continuous_with_interval(pred: np.ndarray, obs: np.ndarray,
                             groups: List[np.ndarray], metric: str = "rmse",
                             iters: int = DEFAULT_ITERS,
                             seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    point = compute_continuous_metrics(pred, obs)
    rng = np.random.RandomState(seed)
    samples = np.empty(iters)
    for i in range(iters):
        idx = _resample(groups, rng)
        samples[i] = compute_continuous_metrics(pred[idx], obs[idx])[metric]
    out = {"value": point[metric]}
    out["ci95"] = [round(float(np.percentile(samples, 2.5)), 4),
                   round(float(np.percentile(samples, 97.5)), 4)]
    return out


def stratified_categorical(df, systems: Dict[str, np.ndarray], threshold: float,
                           stratify_col: str, system_order: Optional[List[str]] = None,
                           interval_iters: int = 200,
                           paired: Optional[tuple] = None) -> Dict[str, Any]:
    """
    One categorical table per stratum of `stratify_col`, for several systems.

    `paired=(A, B)` additionally reports the A-minus-B improvement with a
    day-block interval inside each stratum.
    """
    obs = df["obs_rain_max"].to_numpy(float)
    days = df["date"].to_numpy()
    labels = df[stratify_col].to_numpy()
    names = list(system_order) if system_order else list(systems.keys())

    table: Dict[str, Any] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if not mask.any():
            continue
        sub_days = days[mask]
        groups = day_blocks(sub_days)
        counts = {
            "heavy": int((obs[mask] >= 64.5).sum()),
            "very_heavy": int((obs[mask] >= 115.6).sum()),
            "extremely_heavy": int((obs[mask] >= 204.5).sum()),
        }
        events = int((obs[mask] >= threshold).sum())
        entry: Dict[str, Any] = {
            "sample_count": int(mask.sum()),
            "day_count": int(len(groups)),
            "events": events,
            "event_counts": counts,
            "reliability": reliability_class(events),
            "reportable": events >= MIN_EVENTS,
            "systems": {},
        }
        for name in names:
            entry["systems"][name] = score(systems[name][mask], obs[mask], threshold)
            entry["systems"][name]["rmse"] = compute_continuous_metrics(
                systems[name][mask], df["obs_rain_mean"].to_numpy(float)[mask])["rmse"]
            entry["systems"][name]["bias"] = compute_continuous_metrics(
                systems[name][mask], df["obs_rain_mean"].to_numpy(float)[mask])["bias"]

        if paired:
            a, b = paired
            if entry["reportable"]:
                entry["improvement"] = paired_delta(
                    systems[a][mask], systems[b][mask], obs[mask], threshold, groups,
                    metric="csi", iters=interval_iters)
            else:
                entry["improvement"] = {
                    "suppressed": True,
                    "reason": f"only {events} events (< {MIN_EVENTS}); score not reportable",
                }
        table[str(label)] = entry
    return table


def contingency_summary(pred: np.ndarray, obs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compact contingency block for the PDF/JSON without intervals."""
    table = compute_contingency_table(pred, obs, threshold)
    out = compute_dichotomous_metrics(table)
    out["event_count"] = int(table["hits"] + table["misses"])
    out["reliability"] = reliability_class(out["event_count"])
    return out
