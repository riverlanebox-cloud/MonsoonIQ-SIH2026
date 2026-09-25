"""
MonsoonIQ Verification Metrics Engine.
Implements:
1. Continuous: RMSE, Bias, MAE, Pearson Correlation
2. Dichotomous (Contingency Table): POD, FAR, CSI, Frequency Bias, ETS (Gilbert Skill Score)
3. Spatial: Fractions Skill Score (FSS) at neighborhood scales (1, 3, 5, 9 grid cells)
4. Probabilistic: Brier Score, ROC curve & AUC, Reliability diagram bins
5. Bootstrap Confidence Intervals (95% CI)
"""

import numpy as np
import scipy.signal
import scipy.stats
from typing import Dict, Any, List, Tuple, Optional


def compute_continuous_metrics(forecast: np.ndarray, observed: np.ndarray) -> Dict[str, float]:
    """Calculate continuous forecast accuracy metrics."""
    f = np.asarray(forecast, dtype=float).ravel()
    o = np.asarray(observed, dtype=float).ravel()
    valid = np.isfinite(f) & np.isfinite(o)
    f, o = f[valid], o[valid]

    if len(f) == 0:
        return {"rmse": 0.0, "mae": 0.0, "bias": 0.0, "correlation": 0.0}

    diff = f - o
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    if np.std(f) > 1e-6 and np.std(o) > 1e-6:
        corr = float(np.corrcoef(f, o)[0, 1])
    else:
        corr = 0.0

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "bias": round(bias, 3),
        "correlation": round(corr, 3)
    }


def compute_contingency_table(forecast: np.ndarray, observed: np.ndarray, threshold: float) -> Dict[str, int]:
    """Construct 2x2 contingency table for a precipitation threshold."""
    f = np.asarray(forecast, dtype=float).ravel()
    o = np.asarray(observed, dtype=float).ravel()
    valid = np.isfinite(f) & np.isfinite(o)
    f, o = f[valid], o[valid]

    f_yes = (f >= threshold)
    o_yes = (o >= threshold)

    a = int(np.sum(f_yes & o_yes))       # Hits
    b = int(np.sum(f_yes & ~o_yes))      # False Alarms
    c = int(np.sum(~f_yes & o_yes))      # Misses
    d = int(np.sum(~f_yes & ~o_yes))     # Correct Negatives

    return {"hits": a, "false_alarms": b, "misses": c, "correct_negatives": d, "total": a + b + c + d}


def compute_dichotomous_metrics(table: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate POD, FAR, CSI, FBIAS, and ETS from a 2x2 contingency table.
    """
    a = table["hits"]
    b = table["false_alarms"]
    c = table["misses"]
    d = table["correct_negatives"]
    total = table["total"]

    # Probability of Detection (Hit Rate)
    pod = float(a / (a + c)) if (a + c) > 0 else 0.0

    # False Alarm Ratio
    far = float(b / (a + b)) if (a + b) > 0 else 0.0

    # Critical Success Index (Threat Score)
    csi = float(a / (a + b + c)) if (a + b + c) > 0 else 0.0

    # Frequency Bias
    fbias = float((a + b) / (a + c)) if (a + c) > 0 else (1.0 if (a + b) == 0 else 99.0)

    # Equitable Threat Score (Gilbert Skill Score)
    # a_r = hits expected by random chance
    if total > 0:
        a_r = ((a + b) * (a + c)) / float(total)
        denom = (a + b + c - a_r)
        ets = float((a - a_r) / denom) if denom != 0 else 0.0
    else:
        ets = 0.0

    return {
        "hits": a,
        "false_alarms": b,
        "misses": c,
        "correct_negatives": d,
        "event_count": a + c,
        "pod": round(pod, 4),
        "far": round(far, 4),
        "csi": round(csi, 4),
        "frequency_bias": round(fbias, 3),
        "ets": round(ets, 4)
    }


def compute_fractions_skill_score_2d(forecast_grid: np.ndarray, observed_grid: np.ndarray,
                                     threshold: float, window_size: int) -> float:
    """
    Compute Fractions Skill Score (FSS) on a 2D spatial grid for neighborhood window_size (e.g. 1, 3, 5, 9).
    FSS = 1 - (MSE_k / MSE_k_ref).
    """
    f_bin = (forecast_grid >= threshold).astype(float)
    o_bin = (observed_grid >= threshold).astype(float)

    if window_size == 1:
        p_f = f_bin
        p_o = o_bin
    else:
        kernel = np.ones((window_size, window_size), dtype=float) / (window_size * window_size)
        p_f = scipy.signal.convolve2d(f_bin, kernel, mode="same", boundary="symm")
        p_o = scipy.signal.convolve2d(o_bin, kernel, mode="same", boundary="symm")

    mse = np.mean((p_f - p_o) ** 2)
    mse_ref = np.mean(p_f ** 2 + p_o ** 2)

    if mse_ref < 1e-8:
        # Both fields are completely zero (no event anywhere)
        return 1.0 if mse < 1e-8 else 0.0

    fss = 1.0 - (mse / mse_ref)
    return float(np.clip(fss, 0.0, 1.0))


def compute_reliability_diagram(probabilities: np.ndarray, observations: np.ndarray,
                                n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute reliability curve coordinates and sharpness histogram.
    """
    p = np.asarray(probabilities, dtype=float).ravel()
    o = np.asarray(observations, dtype=float).ravel()
    valid = np.isfinite(p) & np.isfinite(o)
    p, o = p[valid], o[valid]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_forecast_probs = []
    observed_frequencies = []
    sample_counts = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        mask = (p >= low) & (p < high) if i < n_bins - 1 else (p >= low) & (p <= high)
        count = int(np.sum(mask))
        sample_counts.append(count)

        if count > 0:
            mean_forecast_probs.append(float(np.mean(p[mask])))
            observed_frequencies.append(float(np.mean(o[mask])))
        else:
            mean_forecast_probs.append(float((low + high) / 2.0))
            observed_frequencies.append(float((low + high) / 2.0))

    brier_score = float(np.mean((p - o) ** 2))

    return {
        "bin_edges": [round(float(b), 2) for b in bin_edges],
        "mean_forecast_probs": [round(x, 4) for x in mean_forecast_probs],
        "observed_frequencies": [round(x, 4) for x in observed_frequencies],
        "sample_counts": sample_counts,
        "brier_score": round(brier_score, 4)
    }


def compute_roc_curve(probabilities: np.ndarray, observations: np.ndarray,
                      n_thresholds: int = 50) -> Dict[str, Any]:
    """Compute empirical ROC curve coordinates and Area Under Curve (AUC)."""
    p = np.asarray(probabilities, dtype=float).ravel()
    o = np.asarray(observations, dtype=int).ravel()
    valid = np.isfinite(p) & np.isfinite(o)
    p, o = p[valid], o[valid]

    positives = np.sum(o == 1)
    negatives = np.sum(o == 0)

    if positives == 0 or negatives == 0:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    tpr_list = []
    fpr_list = []

    for t in sorted(thresholds, reverse=True):
        pred_pos = (p >= t)
        tp = np.sum(pred_pos & (o == 1))
        fp = np.sum(pred_pos & (o == 0))
        tpr = tp / positives
        fpr = fp / negatives
        tpr_list.append(float(tpr))
        fpr_list.append(float(fpr))

    # Add endpoints
    fpr_arr = np.array([0.0] + fpr_list + [1.0])
    tpr_arr = np.array([0.0] + tpr_list + [1.0])

    # Trapezoidal AUC
    order = np.argsort(fpr_arr)
    auc = float(np.trapezoid(tpr_arr[order], fpr_arr[order]))

    return {
        "fpr": [round(float(x), 4) for x in fpr_arr[order]],
        "tpr": [round(float(y), 4) for y in tpr_arr[order]],
        "auc": round(auc, 4)
    }


def compute_bootstrap_ci(metric_fn, forecast: np.ndarray, observed: np.ndarray,
                         n_bootstraps: int = 300, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for any verification metric."""
    rng = np.random.RandomState(seed)
    N = len(forecast)
    boot_scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, N, size=N)
        f_sample = forecast[idx]
        o_sample = observed[idx]
        val = metric_fn(f_sample, o_sample)
        if np.isfinite(val):
            boot_scores.append(val)

    if not boot_scores:
        return (0.0, 0.0)

    lower_p = (1.0 - ci) / 2.0 * 100.0
    upper_p = (1.0 + ci) / 2.0 * 100.0
    low = float(np.percentile(boot_scores, lower_p))
    high = float(np.percentile(boot_scores, upper_p))
    return round(low, 4), round(high, 4)
