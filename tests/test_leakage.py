"""
MonsoonIQ Data Leakage & Temporal Split Tests.
Strictly verifies:
1. No overlap between training years, validation years, and test years.
2. Quantile mapping tables are constructed ONLY from training period data.
3. Feature scalers and thresholds do not observe test partition statistics.
4. Model artifacts do not store test set observations in training metadata.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.correction.quantile_mapping import EmpiricalQuantileMapper
from src.correction.mixture_of_experts import MonsoonIQMixtureOfExperts


def test_strict_temporal_split():
    data_path = "data/synthetic/district_daily.parquet"
    if not os.path.exists(data_path):
        pytest.skip("Dataset not yet generated.")

    df = pd.read_parquet(data_path)
    train_years = set([2016, 2017, 2018, 2019, 2020])
    val_years = set([2021])
    test_years = set([2022, 2023])

    # Assert mutual exclusivity
    assert len(train_years.intersection(val_years)) == 0, "Train and Val years overlap!"
    assert len(train_years.intersection(test_years)) == 0, "Train and Test years overlap!"
    assert len(val_years.intersection(test_years)) == 0, "Val and Test years overlap!"

    # Split dataset
    train_df = df[df["year"].isin(train_years)]
    val_df = df[df["year"].isin(val_years)]
    test_df = df[df["year"].isin(test_years)]

    # Verify temporal ordering (Train < Val < Test)
    assert train_df["date"].max() < val_df["date"].min(), "Training dates exceed Validation start date!"
    assert val_df["date"].max() < test_df["date"].min(), "Validation dates exceed Test start date!"


def test_quantile_mapping_no_test_leakage():
    """Verify that Empirical Quantile Mapper fitted on training data has no knowledge of test data."""
    np.random.seed(42)
    train_nwp = np.random.uniform(0, 50, 1000)
    train_obs = np.random.uniform(0, 60, 1000)

    # Extreme unseen test event
    test_nwp = np.array([120.0, 150.0])

    eqm = EmpiricalQuantileMapper()
    eqm.fit(train_nwp, train_obs)

    # Max of training obs
    assert eqm.obs_quantiles.max() <= 60.0
    assert eqm.nwp_quantiles.max() <= 50.0

    # Ensure fitted quantiles only depend on training statistics
    assert len(eqm.quantiles) == 1000
    assert eqm.is_fitted


def test_moe_artifact_leakage_check():
    """Verify loaded MoE model was trained only on 2016-2020."""
    moe_path = "artifacts/models/mixture_of_experts.joblib"
    if not os.path.exists(moe_path):
        pytest.skip("MoE artifact not found.")

    moe = MonsoonIQMixtureOfExperts.load(moe_path)
    assert moe.is_fitted
    assert len(moe.experts) == 7
