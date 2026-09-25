"""
MonsoonIQ verification evaluator.

Produces, on a strictly held-out period:

  * continuous skill (RMSE / MAE / bias / correlation) for every system
  * categorical skill at the IMD thresholds (POD, FAR, CSI, ETS, frequency bias)
  * the same, stratified by regime, by geographic zone and by forecast lead time
  * every stratum carries its event count and a reliability class, and thin
    strata are marked `reportable: false` rather than silently printed as 0.000
  * day-block bootstrap intervals (whole calendar days resampled, so the
    spatio-temporal correlation of the sample is respected)
  * a paired regime-value ablation: classifier weights vs uniform weights vs a
    regime-agnostic model fitted on the same predictors
  * Fractions Skill Score at neighbourhood windows on the sampled grid
  * a significance statement that is *derived* from the computed intervals

Nothing here writes a claim that the numbers do not support.
"""

import os
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.verification.metrics import (
    compute_continuous_metrics,
    compute_contingency_table,
    compute_dichotomous_metrics,
    compute_reliability_diagram,
    compute_roc_curve,
)
from src.verification.stratified import (
    MIN_EVENTS,
    day_blocks,
    paired_delta,
    reliability_class,
    score,
    score_with_interval,
    stratified_categorical,
)

logger = logging.getLogger(__name__)

SYSTEMS = ("raw_nwp", "global_qm", "global_lgb", "monsooniq")
SYSTEM_LABELS = {
    "raw_nwp": "Raw NWP",
    "global_qm": "Quantile mapping (regime-agnostic)",
    "global_lgb": "Gradient boosting (regime-agnostic)",
    "monsooniq": "MonsoonIQ (regime-aware)",
}


def _archive_fingerprint(path: str) -> Optional[str]:
    """First 16 hex chars of the archive's SHA-256, or None if it is missing.

    The generator is bit-reproducible inside one environment, but last-bit
    differences in NumPy/BLAS across environments move the archive by ~1e-13.
    Recording the fingerprint with the results means any artifact can be traced
    back to the exact archive it was scored against.
    """
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def detect_provenance(metadata_path: str = "data/synthetic/dataset_metadata.json",
                      archive_path: str = "data/synthetic/district_daily.parquet") -> Dict[str, Any]:
    """Read provenance from the dataset's own metadata instead of hard-coding it."""
    fingerprint = _archive_fingerprint(archive_path)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "provenance": meta.get("dataset_type", "UNKNOWN_DATASET"),
                "is_synthetic": "SYNTHETIC" in str(meta.get("dataset_type", "")).upper(),
                "seed": meta.get("seed"),
                "domain": meta.get("domain"),
                "districts": meta.get("districts_count"),
                "total_days": meta.get("total_days"),
                "archive_sha256_16": fingerprint,
                "note": meta.get("provenance_note", ""),
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not read dataset metadata: %s", exc)
    return {"provenance": "UNKNOWN_DATASET", "is_synthetic": None,
            "archive_sha256_16": fingerprint}


class VerificationEvaluator:
    """Evaluates forecasts across systems, regimes, zones, leads and scales."""

    THRESHOLDS = [2.5, 15.6, 64.5, 115.6, 204.5]
    REGIME_NAMES = {
        1: "Active Monsoon", 2: "Break Monsoon", 3: "Monsoon Low/Depression",
        4: "Orographic", 5: "Coastal", 6: "Western Disturbance", 7: "Weak/Normal",
    }

    def __init__(self, metrics_dir: str = "artifacts/metrics",
                 plots_dir: str = "artifacts/plots"):
        self.metrics_dir = metrics_dir
        self.plots_dir = plots_dir
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    # ------------------------------------------------------------------ public
    def evaluate_test_set(self, test_df: pd.DataFrame,
                          predictions: Dict[str, np.ndarray],
                          probabilities: Optional[Dict[str, np.ndarray]] = None,
                          moe=None,
                          lead_bundle=None,
                          regime_posteriors: Optional[np.ndarray] = None,
                          grid_npz_path: str = "data/synthetic/grid_feature_samples.npz",
                          interval_iters: int = 300) -> Dict[str, Any]:
        y = test_df["obs_rain_mean"].to_numpy(float)
        y_max = test_df["obs_rain_max"].to_numpy(float)
        regimes = test_df["regime"].to_numpy(int)
        zones = test_df["zone"].to_numpy()
        days = test_df["date"].to_numpy()
        groups = day_blocks(days)

        continuous = {s: compute_continuous_metrics(predictions[s], y) for s in SYSTEMS}

        # --- categorical at every IMD threshold -------------------------------
        threshold_metrics: Dict[str, Any] = {}
        for t in self.THRESHOLDS:
            key = f"{t:.1f}"
            threshold_metrics[key] = {s: score(predictions[s], y_max, t) for s in SYSTEMS}
            threshold_metrics[key]["event_count"] = int((y_max >= t).sum())
            threshold_metrics[key]["reliability"] = reliability_class(
                threshold_metrics[key]["event_count"])

        # --- regime stratification with gating and paired improvement --------
        regime_key = test_df.copy()
        regime_key["_regime_name"] = [self.REGIME_NAMES[r] for r in regimes]
        regime_stratified = stratified_categorical(
            regime_key, predictions, 64.5, stratify_col="_regime_name",
            system_order=list(SYSTEMS), interval_iters=interval_iters,
            paired=("monsooniq", "raw_nwp"))

        # legacy shape kept for existing consumers (UI/API/tests)
        regime_breakdown = {}
        for name, entry in regime_stratified.items():
            regime_breakdown[name] = {
                "sample_count": entry["sample_count"],
                "day_count": entry["day_count"],
                "heavy_events": entry["events"],
                "very_heavy_events": entry["event_counts"]["very_heavy"],
                "reliability": entry["reliability"],
                "reportable": entry["reportable"],
                "rmse": {s: entry["systems"][s]["rmse"] for s in SYSTEMS},
                "heavy_csi": {s: entry["systems"][s]["csi"] for s in SYSTEMS},
                "heavy_ets": {s: entry["systems"][s]["ets"] for s in SYSTEMS},
                "heavy_pod": {s: entry["systems"][s]["pod"] for s in SYSTEMS},
                "heavy_far": {s: entry["systems"][s]["far"] for s in SYSTEMS},
                "improvement_vs_raw_nwp": entry.get("improvement"),
            }

        # --- geographic zones (deterministic order) --------------------------
        zone_breakdown = {}
        for z in sorted(set(zones.tolist())):
            mask = zones == z
            zone_breakdown[z] = {
                "sample_count": int(mask.sum()),
                "heavy_events": int((y_max[mask] >= 64.5).sum()),
                "reliability": reliability_class(int((y_max[mask] >= 64.5).sum())),
                "rmse": {s: compute_continuous_metrics(predictions[s][mask], y[mask])["rmse"]
                         for s in SYSTEMS},
                "bias": {s: compute_continuous_metrics(predictions[s][mask], y[mask])["bias"]
                         for s in SYSTEMS},
                "heavy_csi": {s: score(predictions[s][mask], y_max[mask], 64.5)["csi"]
                              for s in SYSTEMS},
            }

        # --- lead time -------------------------------------------------------
        lead_time_breakdown: Dict[str, Any] = {"available": False}
        if lead_bundle is not None:
            try:
                lead_time_breakdown = {
                    "available": True,
                    "note": ("Each lead is a separately fitted correction system; "
                             "comparisons are like-for-like within a lead."),
                    "leads": lead_bundle.verification_matrix(
                        test_df, regime_probs=regime_posteriors),
                }
            except Exception as exc:  # pragma: no cover
                logger.warning("Lead-time verification failed: %s", exc)
                lead_time_breakdown = {"available": False, "reason": str(exc)}

        # --- probabilistic ---------------------------------------------------
        prob_analysis: Dict[str, Any] = {}
        if probabilities:
            name_map = {"heavy": 64.5, "very_heavy": 115.6, "extremely_heavy": 204.5}
            for key, thr in name_map.items():
                if key not in probabilities:
                    continue
                obs_bin = (y_max >= thr).astype(int)
                ev = int(obs_bin.sum())
                block: Dict[str, Any] = {
                    "threshold_mm": thr,
                    "event_count": ev,
                    "reliability": reliability_class(ev),
                    "brier_score": round(float(np.mean((probabilities[key] - obs_bin) ** 2)), 5),
                    "mean_forecast_probability": round(float(np.mean(probabilities[key])), 4),
                    "observed_frequency": round(float(obs_bin.mean()), 4),
                }
                if ev >= MIN_EVENTS:
                    block["reliability_diagram"] = compute_reliability_diagram(
                        probabilities[key], obs_bin)
                    block["roc"] = compute_roc_curve(probabilities[key], obs_bin)
                else:
                    block["roC_suppressed"] = (
                        f"{ev} events (< {MIN_EVENTS}); reliability and ROC not reportable")
                prob_analysis[key] = block

        # --- spatial (FSS) ---------------------------------------------------
        grid_fss: Dict[str, Any] = {"available": False}
        if moe is not None and os.path.exists(grid_npz_path):
            try:
                from src.verification.grid_verification import run_grid_verification
                grid_fss = run_grid_verification(moe, test_df.assign(
                    year=test_df["year"]), test_df, npz_path=grid_npz_path)
            except Exception as exc:  # pragma: no cover
                logger.warning("Grid FSS failed: %s", exc)
                grid_fss = {"available": False, "reason": str(exc)}

        # --- regime value ablation -------------------------------------------
        regime_value: Dict[str, Any] = {"available": False}
        if moe is not None:
            try:
                from src.verification.regime_value import run_regime_value_analysis
                regime_value = run_regime_value_analysis(
                    moe, test_df, posteriors=regime_posteriors, iters=interval_iters)
                regime_value["available"] = True
            except Exception as exc:  # pragma: no cover
                logger.warning("Regime-value analysis failed: %s", exc)
                regime_value = {"available": False, "reason": str(exc)}

        # --- intervals and the derived significance statement -----------------
        comparisons = {
            "rmse_vs_raw_nwp": self._rmse_comparison(predictions, y, groups, "raw_nwp", interval_iters),
            "rmse_vs_global_lgb": self._rmse_comparison(predictions, y, groups, "global_lgb", interval_iters),
            "heavy_csi_vs_raw_nwp": paired_delta(
                predictions["monsooniq"], predictions["raw_nwp"], y_max, 64.5, groups,
                metric="csi", iters=interval_iters),
            "heavy_csi_vs_global_lgb": paired_delta(
                predictions["monsooniq"], predictions["global_lgb"], y_max, 64.5, groups,
                metric="csi", iters=interval_iters),
        }
        confidence_intervals = {
            "rmse": {
                "raw_nwp": {"val": continuous["raw_nwp"]["rmse"],
                            "ci_95": comparisons["rmse_vs_raw_nwp"]["b_ci95"]},
                "monsooniq": {"val": continuous["monsooniq"]["rmse"],
                              "ci_95": comparisons["rmse_vs_raw_nwp"]["a_ci95"]},
                "method": "day-block bootstrap (whole calendar days resampled)",
                "statistically_significant": comparisons["rmse_vs_raw_nwp"]["significant_95"],
            },
            "heavy_csi": {
                "raw_nwp": {"val": threshold_metrics["64.5"]["raw_nwp"]["csi"]},
                "monsooniq": {"val": threshold_metrics["64.5"]["monsooniq"]["csi"]},
                "method": "day-block bootstrap (whole calendar days resampled)",
                "statistically_significant": comparisons["heavy_csi_vs_raw_nwp"]["significant_95"],
            },
        }

        provenance = detect_provenance()
        summary = {
            "data_provenance": provenance["provenance"],
            "provenance_detail": provenance,
            "verification_period": {
                "test_years": sorted(set(test_df["year"].astype(int).tolist())),
                "district_days": int(len(test_df)),
                "calendar_days": int(len(groups)),
                "districts": int(test_df["district_id"].nunique()),
            },
            "test_sample_count": int(len(test_df)),
            "test_years": sorted(set(test_df["year"].astype(int).tolist())),
            "systems": {s: SYSTEM_LABELS[s] for s in SYSTEMS},
            "continuous_metrics": continuous,
            "threshold_metrics": threshold_metrics,
            "regime_breakdown": regime_breakdown,
            "regime_stratified": regime_stratified,
            "zone_breakdown": zone_breakdown,
            "lead_time_breakdown": lead_time_breakdown,
            "confidence_intervals": confidence_intervals,
            "paired_comparisons": comparisons,
            "probabilistic_verification": prob_analysis,
            "grid_fss": grid_fss,
            "regime_value": regime_value,
            "significance_statement": self._significance_statement(comparisons, regime_value),
            "scorecard": self._scorecard(continuous, threshold_metrics, comparisons, regime_value),
        }

        self._save(summary, test_df, predictions)
        return summary

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _rmse_comparison(predictions, y, groups, baseline, iters) -> Dict[str, Any]:
        rng = np.random.RandomState(20260922)
        a = predictions["monsooniq"]
        b = predictions[baseline]
        deltas = np.empty(iters)
        a_vals = np.empty(iters)
        b_vals = np.empty(iters)
        for i in range(iters):
            picks = rng.randint(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[j] for j in picks])
            a_vals[i] = compute_continuous_metrics(a[idx], y[idx])["rmse"]
            b_vals[i] = compute_continuous_metrics(b[idx], y[idx])["rmse"]
            deltas[i] = a_vals[i] - b_vals[i]
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        return {
            "metric": "rmse",
            "baseline": baseline,
            "a_value": compute_continuous_metrics(a, y)["rmse"],
            "b_value": compute_continuous_metrics(b, y)["rmse"],
            "delta": round(float(np.mean(a_vals) - np.mean(b_vals)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "a_ci95": [round(float(np.percentile(a_vals, 2.5)), 4),
                       round(float(np.percentile(a_vals, 97.5)), 4)],
            "b_ci95": [round(float(np.percentile(b_vals, 2.5)), 4),
                       round(float(np.percentile(b_vals, 97.5)), 4)],
            "significant_95": bool(lo > 0 or hi < 0),
            "favours": "monsooniq" if hi < 0 else ("baseline" if lo > 0 else "neither"),
        }

    @staticmethod
    def _significance_statement(comparisons: Dict[str, Any],
                                regime_value: Dict[str, Any]) -> Dict[str, Any]:
        """Say only what the intervals support."""
        vs_raw_rmse = comparisons["rmse_vs_raw_nwp"]
        vs_raw_csi = comparisons["heavy_csi_vs_raw_nwp"]
        vs_lgb_csi = comparisons["heavy_csi_vs_global_lgb"]
        vs_lgb_rmse = comparisons["rmse_vs_global_lgb"]

        claims = [
            {
                "claim": "Post-processing reduces RMSE relative to the raw model",
                "supported": bool(vs_raw_rmse["significant_95"]),
                "evidence": f"ΔRMSE {vs_raw_rmse['delta']:+.3f} mm/day, 95% CI "
                            f"[{vs_raw_rmse['ci95'][0]:+.3f}, {vs_raw_rmse['ci95'][1]:+.3f}]",
                "verified_on": "synthetic held-out period",
            },
            {
                "claim": "Post-processing improves the heavy-rainfall categorical score "
                         "relative to the raw model",
                "supported": bool(vs_raw_csi["significant_95"] and vs_raw_csi["delta"] > 0),
                "evidence": f"ΔCSI {vs_raw_csi['delta']:+.4f}, 95% CI "
                            f"[{vs_raw_csi['ci95'][0]:+.4f}, {vs_raw_csi['ci95'][1]:+.4f}]",
                "verified_on": "synthetic held-out period",
            },
            {
                "claim": "Regime conditioning beats a regime-agnostic model fitted on the "
                         "same predictors (heavy-rainfall categorical score)",
                "supported": bool(vs_lgb_csi["significant_95"] and vs_lgb_csi["delta"] > 0),
                "evidence": f"ΔCSI {vs_lgb_csi['delta']:+.4f}, 95% CI "
                            f"[{vs_lgb_csi['ci95'][0]:+.4f}, {vs_lgb_csi['ci95'][1]:+.4f}]",
                "verified_on": "synthetic held-out period",
            },
            {
                "claim": "Regime conditioning beats a regime-agnostic model on RMSE",
                "supported": bool(vs_lgb_rmse["significant_95"] and vs_lgb_rmse["favours"] == "monsooniq"),
                "evidence": f"ΔRMSE {vs_lgb_rmse['delta']:+.3f} mm/day, 95% CI "
                            f"[{vs_lgb_rmse['ci95'][0]:+.3f}, {vs_lgb_rmse['ci95'][1]:+.3f}]",
                "verified_on": "synthetic held-out period",
            },
        ]
        unsupported = [c["claim"] for c in claims if not c["supported"]]
        return {
            "claims": claims,
            "headline": ("All evaluated claims are supported on the synthetic held-out period."
                         if not unsupported else
                         "Not all claims are supported: " + "; ".join(unsupported)),
            "caveat": ("Intervals resample whole calendar days. All comparisons are against a "
                       "synthetic archive, so supported claims demonstrate that the pipeline "
                       "is internally consistent - not operational skill."),
        }

    @staticmethod
    def _scorecard(continuous, threshold_metrics, comparisons, regime_value) -> Dict[str, Any]:
        raw_csi = threshold_metrics["64.5"]["raw_nwp"]["csi"]
        moe_csi = threshold_metrics["64.5"]["monsooniq"]["csi"]
        lgb_csi = threshold_metrics["64.5"]["global_lgb"]["csi"]
        cards = [
            {"label": "RMSE", "raw": continuous["raw_nwp"]["rmse"],
             "baseline": continuous["global_lgb"]["rmse"],
             "corrected": continuous["monsooniq"]["rmse"], "unit": "mm/day",
             "lower_is_better": True},
            {"label": "Heavy-rain CSI (≥64.5 mm)", "raw": raw_csi,
             "baseline": lgb_csi, "corrected": moe_csi, "unit": "", "lower_is_better": False},
            {"label": "Heavy-rain ETS (≥64.5 mm)",
             "raw": threshold_metrics["64.5"]["raw_nwp"]["ets"],
             "baseline": threshold_metrics["64.5"]["global_lgb"]["ets"],
             "corrected": threshold_metrics["64.5"]["monsooniq"]["ets"],
             "unit": "", "lower_is_better": False},
            {"label": "Very heavy-rain POD (≥115.6 mm)",
             "raw": threshold_metrics["115.6"]["raw_nwp"]["pod"],
             "baseline": threshold_metrics["115.6"]["global_lgb"]["pod"],
             "corrected": threshold_metrics["115.6"]["monsooniq"]["pod"],
             "unit": "", "lower_is_better": False},
        ]
        for card in cards:
            denom = card["baseline"] if card["baseline"] else None
            card["vs_raw_pct"] = (round(100 * (card["corrected"] - card["raw"]) / card["raw"], 1)
                                  if card["raw"] else None)
            card["vs_baseline_pct"] = (round(100 * (card["corrected"] - card["baseline"]) / denom, 1)
                                       if denom else None)
        return {
            "cards": cards,
            "honest_baseline": ("global_lgb",
                                "gradient boosting on the same predictors without regime information"),
            "regime_open_question": regime_value.get("headline", {}),
        }

    def _save(self, summary: Dict[str, Any], test_df: pd.DataFrame,
              predictions: Dict[str, np.ndarray]) -> None:
        with open(os.path.join(self.metrics_dir, "verification_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved verification summary to %s/verification_summary.json", self.metrics_dir)

        # Heavy-event view kept as its own artifact: it is what the sponsor asks
        # the team to show first.
        heavy = {"threshold_mm": 64.5, "provenance": summary["data_provenance"],
                 "systems": summary["systems"],
                 "overall_by_system": summary["threshold_metrics"]["64.5"],
                 "by_regime": {name: {
                     "sample_count": e["sample_count"],
                     "day_count": e["day_count"],
                     "event_count": e["events"],
                     "reliability": e["reliability"],
                     "reportable": e["reportable"],
                     "systems": e["systems"],
                     "improvement_vs_raw_nwp": e.get("improvement"),
                 } for name, e in summary["regime_stratified"].items()},
                 "by_lead_time": {k: v["systems"] for k, v in
                                  summary["lead_time_breakdown"].get("leads", {}).items()}
                 if summary["lead_time_breakdown"].get("available") else {},
                 }
        with open(os.path.join(self.metrics_dir, "heavy_events_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(heavy, f, indent=2)
        logger.info("Saved heavy-rain skill artifact to %s/heavy_events_summary.json", self.metrics_dir)
