"""
MonsoonIQ Empirical Quantile Mapping (EQM) Module.
Fits empirical CDFs on training data only (strictly preventing test leakage).
Handles zero-precipitation mass and extreme-value tail extrapolation.
Provides both Global and Regime-Conditioned Quantile Mappers.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EmpiricalQuantileMapper:
    """Non-parametric empirical quantile mapping bias correction."""

    def __init__(self, n_quantiles: int = 1000, wet_day_threshold: float = 0.1):
        self.n_quantiles = n_quantiles
        self.wet_day_threshold = wet_day_threshold
        self.quantiles = np.linspace(0.0, 1.0, n_quantiles)
        self.nwp_quantiles = None
        self.obs_quantiles = None
        self.is_fitted = False

    def fit(self, nwp_train: np.ndarray, obs_train: np.ndarray):
        """Fit empirical quantile distributions on training data strictly."""
        nwp_clean = np.asarray(nwp_train, dtype=np.float64)
        obs_clean = np.asarray(obs_train, dtype=np.float64)

        # Filter valid finite points
        valid_mask = np.isfinite(nwp_clean) & np.isfinite(obs_clean)
        nwp_valid = nwp_clean[valid_mask]
        obs_valid = obs_clean[valid_mask]

        if len(nwp_valid) < 20:
            logger.warning("Very few samples to fit quantile mapping; using identity mapping.")
            self.nwp_quantiles = self.quantiles * 100.0
            self.obs_quantiles = self.quantiles * 100.0
            self.is_fitted = True
            return self

        # Calculate empirical quantiles
        self.nwp_quantiles = np.percentile(nwp_valid, self.quantiles * 100.0)
        self.obs_quantiles = np.percentile(obs_valid, self.quantiles * 100.0)

        # Enforce strict monotonicity to avoid interpolation singularities
        self.nwp_quantiles = np.maximum.accumulate(self.nwp_quantiles)
        self.obs_quantiles = np.maximum.accumulate(self.obs_quantiles)

        self.is_fitted = True
        return self

    def transform(self, nwp_input: np.ndarray) -> np.ndarray:
        """Apply quantile mapping to correct raw NWP forecasts."""
        if not self.is_fitted:
            raise RuntimeError("EmpiricalQuantileMapper must be fitted before transform.")

        arr = np.asarray(nwp_input, dtype=np.float64)
        flat = arr.ravel()

        # Find empirical cumulative probability of nwp values
        # np.interp with monotonically increasing self.nwp_quantiles
        # Handle unique coordinates
        nwp_q_unique, unique_indices = np.unique(self.nwp_quantiles, return_index=True)
        q_unique = self.quantiles[unique_indices]
        obs_q_unique = self.obs_quantiles[unique_indices]

        # Calculate CDF percentile for each NWP point
        p_vals = np.interp(flat, nwp_q_unique, q_unique, left=0.0, right=1.0)

        # Map percentile to observed distribution
        corrected = np.interp(p_vals, q_unique, obs_q_unique)

        # For extreme values beyond the training NWP maximum, extrapolate proportionally
        max_train_nwp = nwp_q_unique[-1]
        max_train_obs = obs_q_unique[-1]
        extreme_mask = flat > max_train_nwp
        if np.any(extreme_mask) and max_train_nwp > 0:
            ratio = max_train_obs / max_train_nwp
            corrected[extreme_mask] = max_train_obs + (flat[extreme_mask] - max_train_nwp) * ratio

        # Enforce physical non-negativity and wet threshold
        corrected = np.maximum(0.0, corrected)
        corrected[flat < self.wet_day_threshold] = 0.0

        return corrected.reshape(arr.shape)


class RegimeQuantileMapper:
    """Manages individual quantile mappers for each weather regime."""

    def __init__(self, regimes: list = list(range(1, 8))):
        self.regimes = regimes
        self.mappers = {r: EmpiricalQuantileMapper() for r in regimes}

    def fit(self, nwp_train: np.ndarray, obs_train: np.ndarray, regimes_train: np.ndarray):
        """Fit separate empirical quantile mapper for each regime partition."""
        for r in self.regimes:
            mask = (regimes_train == r)
            if np.sum(mask) > 30:
                self.mappers[r].fit(nwp_train[mask], obs_train[mask])
            else:
                logger.warning(f"Regime {r} has insufficient training samples ({np.sum(mask)}); fitting on all data.")
                self.mappers[r].fit(nwp_train, obs_train)
        return self

    def transform_single_regime(self, nwp_input: np.ndarray, regime_id: int) -> np.ndarray:
        """Apply regime-specific quantile mapping."""
        mapper = self.mappers.get(regime_id, self.mappers[7])
        return mapper.transform(nwp_input)
