"""
MonsoonIQ evaluation pipeline.

Runs the full verification suite on the held-out years (2022-2023) and writes:

    artifacts/metrics/verification_summary.json   everything the UI reads
    artifacts/metrics/heavy_events_summary.json   heavy-rain skill, by regime/lead
    artifacts/metrics/regime_value_audit.json     what regime awareness is worth
    artifacts/plots/*.png                         figures embedded in the PDF
    artifacts/reports/*.pdf                       the printable verification report

Usage:  PYTHONPATH=. python src/evaluate.py
"""

import os
import json
import logging
import numpy as np
import pandas as pd

from src.regime.ml_classifier import MLRegimeClassifier
from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
from src.correction.lead_time_models import LeadTimeCorrectionBundle
from src.heavy_rain.heavy_rain_classifier import HeavyRainProbabilityModule
from src.correction.quantile_regressor import QuantileRegressor
from src.verification.evaluator import VerificationEvaluator
from src.verification.report_generator import VerificationReportGenerator

logger = logging.getLogger("MonsoonIQ_Evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TEST_YEARS = [2022, 2023]


def run_evaluation_pipeline(data_path: str = "data/synthetic/district_daily.parquet"):
    logger.info("=== MonsoonIQ evaluation ===")
    df = pd.read_parquet(data_path)
    test_df = df[df["year"].isin(TEST_YEARS)].copy().reset_index(drop=True)
    logger.info("Held-out partition: %d records (%s), %d calendar days",
                len(test_df), TEST_YEARS, test_df["date"].nunique())

    regime_clf = MLRegimeClassifier()
    regime_clf.load()
    moe = MonsoonIQMixtureOfExperts.load()
    hrc = HeavyRainProbabilityModule.load()
    qr = QuantileRegressor.load()

    lead_bundle = None
    if os.path.exists("artifacts/models/lead_time_bundle.joblib"):
        try:
            lead_bundle = LeadTimeCorrectionBundle.load()
            logger.info("Loaded per-lead bundle for leads %s", lead_bundle.available_leads())
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not load lead-time bundle: %s", exc)

    logger.info("Predicting regime posteriors and all benchmark systems ...")
    posteriors = regime_clf.predict_proba(test_df)
    predictions = moe.predict_all_systems(test_df, nwp_col="raw_nwp_d1",
                                          regime_probs=posteriors)
    probabilities = hrc.predict_probabilities(test_df, nwp_col="raw_nwp_d1")
    qr.predict_quantiles(test_df, nwp_col="raw_nwp_d1")

    evaluator = VerificationEvaluator()
    summary = evaluator.evaluate_test_set(
        test_df, predictions, probabilities=probabilities, moe=moe,
        lead_bundle=lead_bundle, regime_posteriors=posteriors)

    # Regime-value ablation gets its own artifact (and its own figure).
    if summary.get("regime_value", {}).get("available"):
        with open("artifacts/metrics/regime_value_audit.json", "w", encoding="utf-8") as f:
            json.dump(summary["regime_value"], f, indent=2)
        _plot_regime_value(summary["regime_value"])

    logger.info("Compiling verification report PDF ...")
    from src.verification import report_generator
    try:
        pdf_path = VerificationReportGenerator().generate_pdf()
    except TypeError:
        pdf_path = VerificationReportGenerator().generate_pdf()

    _print_console_summary(summary)
    return summary


def _plot_regime_value(regime_value):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sweep = regime_value["accuracy_sweep"]
        accs = [s["regime_classifier_accuracy"] for s in sweep]
        moe_csi = [s["moe_heavy_csi"] for s in sweep]
        lgb = sweep[0]["regime_agnostic_lgb_csi"]

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        ax.plot(accs, moe_csi, "o-", color="#1d4ed8", lw=2,
                label="Regime-aware correction (heavy-rain CSI)")
        ax.axhline(lgb, color="#b91c1c", ls="--", lw=1.6,
                   label=f"Regime-agnostic model (CSI {lgb:.3f})")
        raw_csi = regime_value["systems"]["raw_nwp"]["csi"]
        ax.axhline(raw_csi, color="#64748b", ls=":", lw=1.6,
                   label=f"Raw NWP (CSI {raw_csi:.3f})")
        ax.set_xlabel("Regime-classifier accuracy")
        ax.set_ylabel("CSI, rainfall ≥ 64.5 mm/day")
        ax.set_title("Where regime conditioning stops paying\n"
                     "Synthetic held-out period 2022–2023", fontsize=11)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        os.makedirs("artifacts/plots", exist_ok=True)
        fig.savefig("artifacts/plots/regime_value_curve.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.warning("Regime-value figure skipped: %s", exc)


def _print_console_summary(summary):
    cont = summary["continuous_metrics"]
    tm = summary["threshold_metrics"]["64.5"]
    print("\n" + "=" * 96)
    print("  MONSOONIQ VERIFICATION — HELD-OUT PERIOD "
          f"{summary['verification_period']['test_years']} "
          f"({summary['verification_period']['calendar_days']} days, "
          f"{summary['verification_period']['district_days']} district-days)")
    print(f"  Provenance: {summary['data_provenance']}")
    print("=" * 96)
    print(f"{'system':<38}{'RMSE':>8}{'bias':>8}{'CSI':>8}{'ETS':>8}{'POD':>8}{'FAR':>8}")
    print("-" * 96)
    for key, label in summary["systems"].items():
        row = tm[key]
        print(f"{label:<38}{cont[key]['rmse']:>8.3f}{cont[key]['bias']:>+8.3f}"
              f"{row['csi']:>8.3f}{row['ets']:>8.3f}{row['pod']:>8.3f}{row['far']:>8.3f}")
    print("=" * 96)

    rv = summary.get("regime_value", {})
    if rv.get("available"):
        h = rv["headline"]
        cmp_ = rv["comparisons"]["moe_classifier_vs_global_lgb"]
        print("\n  REGIME-VALUE ABLATION (the number that matters)")
        print(f"    gain over raw NWP (CSI)              : {h['gain_over_raw_nwp_csi']:+.4f}")
        print(f"    gain over regime-agnostic model (CSI): "
              f"{h['regime_conditioning_gain_over_regime_agnostic_csi']:+.4f} "
              f"95% CI [{cmp_['ci95'][0]:+.4f}, {cmp_['ci95'][1]:+.4f}] "
              f"-> {'SIGNIFICANT' if h['gain_significant_95'] else 'NOT significant'}")
        print(f"    regime weights alone (uniform blend) : "
              f"{h['experts_alone_gain_over_raw_nwp_csi']:+.4f}")
        print(f"    classifier accuracy needed to break even: {h['crossover_accuracy']}")

    print("\n  CLAIM CHECK")
    for claim in summary["significance_statement"]["claims"]:
        flag = "SUPPORTED    " if claim["supported"] else "NOT SUPPORTED"
        print(f"    [{flag}] {claim['claim']}")
        print(f"                   {claim['evidence']}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    run_evaluation_pipeline()
