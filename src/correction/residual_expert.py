"""
MonsoonIQ Residual Expert Module.
Trains a specialized LightGBM regression model per regime to predict the residual:
Residual = Observed_Rainfall - Quantile_Mapped_Rainfall.
The expert output is: y_expert = max(0, y_qm + predicted_residual).
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RegimeResidualExpert:
    """Specialized machine learning expert for a single weather regime."""

    def __init__(self, regime_id: int, regime_name: str,
                 n_estimators: int = 100, learning_rate: float = 0.05,
                 max_depth: int = 5, num_leaves: int = 25):
        self.regime_id = regime_id
        self.regime_name = regime_name
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + regime_id,
            n_jobs=-1,
            verbose=-1
        )
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_residual_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_residual_val: Optional[np.ndarray] = None):
        """Fit residual regression tree on training data."""
        if len(X_train) < 30:
            logger.warning(f"Regime {self.regime_id} has very few training samples ({len(X_train)}).")

        callbacks = []
        eval_set = None
        if X_val is not None and len(X_val) > 10:
            eval_set = [(X_val, y_residual_val)]
            callbacks = [lgb.early_stopping(stopping_rounds=15, verbose=False)]

        self.model.fit(
            X_train, y_residual_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        self.is_fitted = True
        return self

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the residual adjustment."""
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.model.predict(X)
