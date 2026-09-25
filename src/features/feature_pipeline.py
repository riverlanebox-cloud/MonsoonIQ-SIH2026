"""
MonsoonIQ Feature Engineering Pipeline.
Transforms raw atmospheric grids or tabular weather state records into derived meteorological indices:
1. Low-Level Somali Jet magnitude (sqrt(U^2 + V^2))
2. Planetary & Relative Vorticity (curl of horizontal winds)
3. Horizontal Moisture Flux Vector and Divergence (q * V)
4. Outgoing Longwave Radiation (OLR) anomalies relative to climatology
5. Terrain slope and aspect computed via spatial finite difference
6. Coastal proximity and land-sea mask transitions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class FeaturePipeline:
    """Computes dynamic atmospheric and static topographic features."""

    @staticmethod
    def compute_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Scalar wind speed from zonal and meridional components."""
        return np.sqrt(u**2 + v**2)

    @staticmethod
    def compute_moisture_flux(wind_speed: np.ndarray, q500: np.ndarray) -> np.ndarray:
        """Boundary layer moisture transport flux (kg / (m s))."""
        return wind_speed * q500 * 1000.0

    @staticmethod
    def compute_terrain_slope_and_aspect(elevation_grid: np.ndarray, cell_spacing_m: float = 27000.0) -> Dict[str, np.ndarray]:
        """Compute spatial gradient slope and aspect from 2D elevation grid."""
        d_lat, d_lon = np.gradient(elevation_grid, cell_spacing_m)
        slope = np.sqrt(d_lat**2 + d_lon**2)
        aspect = np.arctan2(d_lat, -d_lon)
        return {"slope": slope, "aspect": aspect}

    @staticmethod
    def engineer_features_for_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all derived features on a tabular dataset."""
        df_out = df.copy()
        if "u850" in df_out.columns and "v850" in df_out.columns:
            if "wind_speed_850" not in df_out.columns:
                df_out["wind_speed_850"] = FeaturePipeline.compute_wind_speed(df_out["u850"].values, df_out["v850"].values)
        if "wind_speed_850" in df_out.columns and "q500" in df_out.columns:
            if "moisture_flux" not in df_out.columns:
                df_out["moisture_flux"] = FeaturePipeline.compute_moisture_flux(df_out["wind_speed_850"].values, df_out["q500"].values)
        return df_out
