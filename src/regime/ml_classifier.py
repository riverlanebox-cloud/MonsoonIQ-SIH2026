"""
MonsoonIQ ML Regime Classifier.
Trains a calibrated LightGBM multiclass classifier to predict soft weather regime probabilities
from atmospheric and topographic predictors.
Provides SHAP feature attribution explainability.
Labels are rule-derived from expert meteorological domain rules.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import shap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class MLRegimeClassifier:
    """Calibrated LightGBM multiclass weather regime classifier with SHAP explainability."""

    FEATURE_COLS = [
        "u850", "v850", "wind_speed_850", "vorticity_850", "mslp_anomaly",
        "q500", "cape", "olr", "olr_anomaly", "moisture_flux", "trough_latitude",
        "elevation", "slope", "dist_coast", "latitude", "longitude"
    ]

    REGIME_NAMES = {
        1: "Active Monsoon",
        2: "Break Monsoon",
        3: "Monsoon Low/Depression",
        4: "Orographic",
        5: "Coastal",
        6: "Western Disturbance",
        7: "Weak/Normal"
    }

    def __init__(self, model_save_path: str = "artifacts/models/regime_classifier.joblib"):
        self.model_save_path = model_save_path
        self.model = None
        self.calibrated_model = None
        self.explainer = None
        self.classes_ = list(range(1, 8))

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str = "regime") -> Dict[str, Any]:
        """
        Train multiclass LightGBM with probability calibration.
        Labels (1-7) mapped to 0-6 for LightGBM, then restored.
        """
        X_train = train_df[self.FEATURE_COLS].copy()
        y_train = (train_df[target_col].astype(int) - 1).values # 0-6 index

        X_val = val_df[self.FEATURE_COLS].copy()
        y_val = (val_df[target_col].astype(int) - 1).values

        logger.info(f"Training ML Regime Classifier on {len(X_train)} samples, validating on {len(X_val)} samples...")

        base_clf = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=6,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        base_clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )

        self.model = base_clf

        # Calibrate probabilities with sigmoid / isotonic on validation set
        try:
            calibrator = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
            calibrator.fit(X_val, y_val)
            self.calibrated_model = calibrator
        except Exception as e:
            logger.warning(f"Calibration on validation set skipped ({e}); using base calibrated tree.")
            self.calibrated_model = base_clf

        # Compute validation metrics
        y_val_pred = self.predict(X_val)
        y_val_original = y_val + 1
        acc = accuracy_score(y_val_original, y_val_pred)
        macro_f1 = f1_score(y_val_original, y_val_pred, average="macro")
        conf_mat = confusion_matrix(y_val_original, y_val_pred, labels=self.classes_)
        class_rep = classification_report(y_val_original, y_val_pred, labels=self.classes_,
                                          target_names=[self.REGIME_NAMES[c] for c in self.classes_],
                                          output_dict=True)

        logger.info(f"Validation Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

        # Initialize SHAP TreeExplainer on a sample background
        bg_sample = X_train.sample(min(200, len(X_train)), random_state=42)
        try:
            self.explainer = shap.TreeExplainer(base_clf)
        except Exception as e:
            logger.warning(f"TreeExplainer initialization note: {e}")

        # Save artifacts
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "feature_cols": self.FEATURE_COLS,
            "classes": self.classes_
        }, self.model_save_path)
        logger.info(f"Saved ML Regime Classifier to {self.model_save_path}")

        return {
            "validation_accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "confusion_matrix": conf_mat.tolist(),
            "classification_report": class_rep
        }

    def load(self):
        """Load trained model from artifact storage."""
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"Model file not found: {self.model_save_path}")
        bundle = joblib.load(self.model_save_path)
        self.model = bundle["model"]
        self.calibrated_model = bundle.get("calibrated_model", self.model)
        self.classes_ = bundle.get("classes", list(range(1, 8)))
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated soft regime probabilities [N, 7]."""
        if self.calibrated_model is None:
            self.load()
        feat_df = X[self.FEATURE_COLS]
        probs = self.calibrated_model.predict_proba(feat_df)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict dominant regime class ID (1-7)."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.array([self.classes_[idx] for idx in pred_indices])

    def explain_sample(self, sample_dict: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Compute SHAP feature attribution values for a single sample."""
        if self.model is None:
            self.load()
        df_row = pd.DataFrame([sample_dict])[self.FEATURE_COLS]
        probs = self.predict_proba(df_row)[0]
        dom_idx = int(np.argmax(probs))
        dom_regime_id = self.classes_[dom_idx]

        attributions = []
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(df_row)
            # shap_values can be list of arrays per class, or 3D array [1, num_features, num_classes]
            if isinstance(shap_values, list):
                class_shap = shap_values[dom_idx][0]
            elif shap_values.ndim == 3:
                class_shap = shap_values[0, :, dom_idx]
            else:
                class_shap = shap_values[0]

            for feat, val, s_val in zip(self.FEATURE_COLS, df_row.iloc[0], class_shap):
                attributions.append({
                    "feature": feat,
                    "value": round(float(val), 4),
                    "shap_contribution": round(float(s_val), 4)
                })

            attributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
            top_attributions = attributions[:top_k]
        else:
            top_attributions = []

        return {
            "predicted_regime_id": dom_regime_id,
            "predicted_regime_name": self.REGIME_NAMES[dom_regime_id],
            "soft_probabilities": {self.REGIME_NAMES[k]: round(float(p), 4) for k, p in zip(self.classes_, probs)},
            "top_feature_attributions": top_attributions
        }
