"""
MonsoonIQ Interactive Walkthrough Notebook Demo.
Demonstrates:
1. Loading district data and models
2. Classifying weather regimes and generating SHAP feature attributions
3. Applying soft-blended Mixture of Experts
4. Predicting calibrated heavy-rain probabilities and uncertainty bands
"""

import os
import sys
import pandas as pd
import numpy as np

# Configure UTF-8 for Devanagari text on Windows terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.regime.ml_classifier import MLRegimeClassifier
from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
from src.heavy_rain.heavy_rain_classifier import HeavyRainProbabilityModule
from src.correction.quantile_regressor import QuantileRegressor
from src.api.advisory import AdvisoryGenerator


def main():
    print("=== MonsoonIQ Python API Demo ===")

    # 1. Load Data
    df = pd.read_parquet("data/synthetic/district_daily.parquet")
    sample_day = df[(df["date"] == "2023-07-15") & (df["district_id"] == "MH_MUM")].iloc[0].to_dict()

    print(f"Sample Record: {sample_day['district_name']} on {sample_day['date']}")
    print(f"Raw NWP Forecast: {sample_day['raw_nwp_d1']:.1f} mm | Observed Rain: {sample_day['obs_rain_mean']:.1f} mm")

    # 2. Regime Prediction & SHAP
    regime_clf = MLRegimeClassifier()
    regime_clf.load()
    explanation = regime_clf.explain_sample(sample_day)

    print(f"\nPredicted Regime: {explanation['predicted_regime_name']}")
    print("Top Feature Drivers:")
    for feat in explanation['top_feature_attributions'][:3]:
        print(f" - {feat['feature']}: value={feat['value']}, SHAP={feat['shap_contribution']:+.3f}")

    # 3. Mixture of Experts Post-Processing
    moe = MonsoonIQMixtureOfExperts.load()
    df_sample = pd.DataFrame([sample_day])
    p_regimes = regime_clf.predict_proba(df_sample)
    preds = moe.predict_all_systems(df_sample, nwp_col="raw_nwp_d1", regime_probs=p_regimes)

    print("\nBenchmark Forecast Comparison:")
    print(f" - Raw NWP:        {preds['raw_nwp'][0]:.1f} mm")
    print(f" - Global QM:      {preds['global_qm'][0]:.1f} mm")
    print(f" - Global LightGBM:{preds['global_lgb'][0]:.1f} mm")
    print(f" - MonsoonIQ (MoE):{preds['monsooniq'][0]:.1f} mm (Bias Delta: {preds['monsooniq'][0] - preds['raw_nwp'][0]:+.1f} mm)")

    # 4. Probabilistic Bands & Heavy Rain Risk
    qr = QuantileRegressor.load()
    quants = qr.predict_quantiles(df_sample, nwp_col="raw_nwp_d1")
    hrc = HeavyRainProbabilityModule.load()
    probs = hrc.predict_probabilities(df_sample, nwp_col="raw_nwp_d1")

    print(f"\nUncertainty Interval: P10={quants['p10'][0]:.1f} mm, P50={quants['p50'][0]:.1f} mm, P90={quants['p90'][0]:.1f} mm")
    print(f"Calibrated Heavy Risk: P(>=64.5mm) = {probs['heavy'][0]*100:.1f}%, P(>=115.6mm) = {probs['very_heavy'][0]*100:.1f}%")

    # 5. Bilingual Advisory
    advisory = AdvisoryGenerator.generate_advisory(
        district_name=sample_day["district_name"],
        state_name=sample_day["state_name"],
        regime_name=explanation["predicted_regime_name"],
        mean_rain=preds["monsooniq"][0],
        max_rain=quants["p90"][0],
        p_heavy=probs["heavy"][0],
        p_very_heavy=probs["very_heavy"][0]
    )

    print(f"\nIMD Alert Level: {advisory['alert_code']} ({advisory['alert_label_en']})")
    print(f"English Farmer Advisory:\n  {advisory['english']['farmers']}")
    print(f"Hindi Farmer Advisory:\n  {advisory['hindi']['farmers']}")


if __name__ == "__main__":
    main()
