"""
MonsoonIQ end-to-end training pipeline.

Strict time-based split throughout (no shuffling, no random splits, no leakage):

    train 2016-2020 | validation 2021 | held-out test 2022-2023

Fits, in order:
  1. Calibrated regime classifier            (soft regime probabilities)
  2. Regime-aware correction + baselines     (Day-1 product)
  3. Quantile regressors                     (P10 / P50 / P90 band)
  4. Calibrated exceedance probabilities     (64.5 / 115.6 / 204.5 mm)
  5. Per-lead correction bundle              (Day 1-5, fitted separately per lead)
  6. Grid-native correction                  (for the 0.5 deg gridded product)

Everything lands in artifacts/models/. Total runtime is about one minute, which
is deliberate: judges can watch a full retrain during the demo.

Usage:  PYTHONPATH=. python src/train.py
"""

import os
import time
import json
import hashlib
import logging
import numpy as np
import pandas as pd

from src.regime.ml_classifier import MLRegimeClassifier
from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts
from src.correction.quantile_regressor import QuantileRegressor
from src.correction.lead_time_models import LeadTimeCorrectionBundle
from src.correction.grid_correction import GridCorrectionModel
from src.heavy_rain.heavy_rain_classifier import HeavyRainProbabilityModule

logger = logging.getLogger("MonsoonIQ_Train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020]
VAL_YEARS = [2021]
TEST_YEARS = [2022, 2023]
MANIFEST = "artifacts/models/training_manifest.json"


def _sha256(path: str, limit: int = 2_000_000) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def run_training_pipeline(data_path: str = "data/synthetic/district_daily.parquet",
                          fit_lead_bundle: bool = True):
    start = time.time()
    logger.info("=== MonsoonIQ training pipeline ===")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run the generator first.")

    df = pd.read_parquet(data_path)
    logger.info("Loaded %d records across years %s", len(df), sorted(df["year"].unique()))

    train_df = df[df["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    val_df = df[df["year"].isin(VAL_YEARS)].copy().reset_index(drop=True)
    test_df = df[df["year"].isin(TEST_YEARS)].copy().reset_index(drop=True)
    logger.info("Splits -> train %d (%s), val %d (%s), test %d (%s)",
                len(train_df), TRAIN_YEARS, len(val_df), VAL_YEARS, len(test_df), TEST_YEARS)

    summary = {"train_samples": len(train_df), "val_samples": len(val_df),
               "test_samples": len(test_df), "split": {"train": TRAIN_YEARS,
                                                       "val": VAL_YEARS, "test": TEST_YEARS}}

    # 1. regime classifier
    logger.info("--- 1/6 regime classifier ---")
    regime_clf = MLRegimeClassifier()
    regime_metrics = regime_clf.train(train_df, val_df, target_col="regime")
    logger.info("Regime classifier validation accuracy: %.4f", regime_metrics["validation_accuracy"])
    summary["regime_classifier"] = {
        "validation_accuracy": regime_metrics["validation_accuracy"],
        "macro_f1": regime_metrics["macro_f1"],
        "labels": "generator latent regime (synthetic)",
    }

    # 2. Day-1 correction + baselines
    logger.info("--- 2/6 regime-aware correction (Day 1) ---")
    moe = MonsoonIQMixtureOfExperts()
    moe.fit(train_df, val_df, nwp_col="raw_nwp_d1", obs_col="obs_rain_mean",
            regime_col="regime")

    # 3. quantile regressors
    logger.info("--- 3/6 quantile regressors ---")
    QuantileRegressor().fit(train_df, obs_col="obs_rain_mean", nwp_col="raw_nwp_d1")

    # 4. exceedance probabilities
    logger.info("--- 4/6 exceedance probability models ---")
    hrc = HeavyRainProbabilityModule()
    hrc_metrics = hrc.fit(train_df, val_df, target_col="obs_rain_max", nwp_col="raw_nwp_d1")
    summary["exceedance_models"] = hrc_metrics

    # 5. per-lead bundle
    if fit_lead_bundle:
        logger.info("--- 5/6 per-lead correction bundle (Day 1-5) ---")
        bundle = LeadTimeCorrectionBundle()
        bundle.fit(train_df, val_df)
        summary["lead_bundle"] = {"leads": bundle.available_leads(),
                                  "detail": bundle.training_summary}
    else:
        summary["lead_bundle"] = {"leads": [], "skipped": True}

    # 6. grid-native correction (needs the sampled grid archive)
    grid_info: Dict[str, Any] = {"skipped": True}
    try:
        from src.correction.grid_correction import GRID_NPZ
        if os.path.exists(GRID_NPZ):
            logger.info("--- 6/6 grid-native correction ---")
            grid_model = GridCorrectionModel().fit(regime_clf)
            grid_info = dict(grid_model.training_meta)
            grid_info["skipped"] = False
        else:
            logger.warning("Grid archive %s missing; skipping grid correction.", GRID_NPZ)
    except Exception as exc:  # pragma: no cover
        logger.warning("Grid correction skipped: %s", exc)
        grid_info = {"skipped": True, "reason": str(exc)}
    summary["grid_correction"] = grid_info

    elapsed = round(time.time() - start, 2)
    summary["elapsed_seconds"] = elapsed
    summary["dataset"] = {"path": data_path, "rows": int(len(df)),
                          "sha256_16": _sha256(data_path)}
    summary["artifacts"] = {
        name: {"sha256_16": _sha256(os.path.join("artifacts/models", f"{name}.joblib")),
               "bytes": os.path.getsize(os.path.join("artifacts/models", f"{name}.joblib"))}
        for name in ("regime_classifier", "mixture_of_experts", "quantile_regressor",
                     "heavy_rain_module", "lead_time_bundle", "grid_correction")
        if os.path.exists(os.path.join("artifacts/models", f"{name}.joblib"))
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Training complete in %.1f s; manifest at %s ===", elapsed, MANIFEST)
    return summary


if __name__ == "__main__":
    run_training_pipeline()
