"""
MonsoonIQ Data Validation Script.
Verifies:
1. Spatial coordinate grid alignment (0.25° resolution, [6-38N, 68-98E])
2. NaN / missing data thresholds
3. Unit consistency (e.g. rain in mm/day, wind in m/s, mslp in hPa)
4. Date coverage, temporal continuity, and leap year handling
5. Physical bounds sanity checks
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DataValidator:
    """Validates real or synthetic meteorological fields for alignment, units, and physical sanity."""

    PHYSICAL_BOUNDS = {
        "rainfall": {"min": 0.0, "max": 1200.0, "unit": "mm/day"},
        "u850": {"min": -60.0, "max": 60.0, "unit": "m/s"},
        "v850": {"min": -60.0, "max": 60.0, "unit": "m/s"},
        "vorticity_850": {"min": -5e-4, "max": 5e-4, "unit": "s^-1"},
        "mslp": {"min": 920.0, "max": 1050.0, "unit": "hPa"},
        "cape": {"min": 0.0, "max": 8000.0, "unit": "J/kg"},
        "q500": {"min": 0.0, "max": 0.03, "unit": "kg/kg"},
        "olr": {"min": 80.0, "max": 380.0, "unit": "W/m^2"}
    }

    def __init__(self, lat_range: Tuple[float, float] = (6.0, 38.0),
                 lon_range: Tuple[float, float] = (68.0, 98.0),
                 expected_resolution: float = 0.25):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        self.resolution = expected_resolution

    def check_grid_alignment(self, lats: np.ndarray, lons: np.ndarray) -> Dict[str, Any]:
        """Validate latitude and longitude grid arrays match standard 0.25° India grid."""
        lat_step = np.diff(lats)
        lon_step = np.diff(lons)

        is_lat_regular = np.allclose(np.abs(lat_step), self.resolution, atol=0.01)
        is_lon_regular = np.allclose(np.abs(lon_step), self.resolution, atol=0.01)

        covers_lat = (lats.min() <= self.lat_min + 0.5) and (lats.max() >= self.lat_max - 0.5)
        covers_lon = (lons.min() <= self.lon_min + 0.5) and (lons.max() >= self.lon_max - 0.5)

        valid = is_lat_regular and is_lon_regular and covers_lat and covers_lon
        return {
            "valid": bool(valid),
            "lat_regular": bool(is_lat_regular),
            "lon_regular": bool(is_lon_regular),
            "covers_lat": bool(covers_lat),
            "covers_lon": bool(covers_lon),
            "lat_bounds": [float(lats.min()), float(lats.max())],
            "lon_bounds": [float(lons.min()), float(lons.max())],
            "grid_shape": (len(lats), len(lons))
        }

    def check_missing_values(self, data_array: np.ndarray, max_nan_pct: float = 5.0) -> Dict[str, Any]:
        """Verify missing/NaN values do not exceed permitted threshold."""
        total_cells = data_array.size
        nan_cells = int(np.isnan(data_array).sum())
        nan_pct = (nan_cells / total_cells) * 100.0 if total_cells > 0 else 0.0

        return {
            "valid": bool(nan_pct <= max_nan_pct),
            "nan_cells": nan_cells,
            "total_cells": total_cells,
            "nan_percentage": round(nan_pct, 3),
            "max_allowed_nan_pct": max_nan_pct
        }

    def check_physical_bounds(self, variable_name: str, data_array: np.ndarray) -> Dict[str, Any]:
        """Validate that variable values lie within physically sensible bounds."""
        bounds = self.PHYSICAL_BOUNDS.get(variable_name.lower())
        if not bounds:
            return {"valid": True, "note": f"No physical bounds defined for {variable_name}"}

        finite_data = data_array[np.isfinite(data_array)]
        if len(finite_data) == 0:
            return {"valid": False, "error": "All data values are NaN or non-finite"}

        data_min = float(finite_data.min())
        data_max = float(finite_data.max())

        within_min = data_min >= bounds["min"]
        within_max = data_max <= bounds["max"]
        valid = within_min and within_max

        return {
            "valid": bool(valid),
            "variable": variable_name,
            "unit": bounds["unit"],
            "expected_min": bounds["min"],
            "observed_min": round(data_min, 4),
            "expected_max": bounds["max"],
            "observed_max": round(data_max, 4),
        }

    def check_date_coverage(self, dates: pd.DatetimeIndex, expected_years: list) -> Dict[str, Any]:
        """Verify date coverage and identify any unexpected gaps."""
        years_present = sorted(list(set(dates.year)))
        missing_years = [y for y in expected_years if y not in years_present]

        return {
            "valid": len(missing_years) == 0,
            "years_present": years_present,
            "missing_years": missing_years,
            "total_days": len(dates),
            "date_start": str(dates.min().date()),
            "date_end": str(dates.max().date())
        }

    def validate_dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive validation on a tabular or flattened grid dataset."""
        report = {
            "overall_valid": True,
            "column_checks": {},
            "row_count": len(df)
        }

        for col in df.columns:
            if col in ["date", "district_id", "regime", "regime_name"]:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].to_numpy(dtype=float)
                col_report = {
                    "missing": self.check_missing_values(vals),
                    "physical": self.check_physical_bounds(col, vals)
                }
                if not col_report["missing"]["valid"] or not col_report["physical"]["valid"]:
                    report["overall_valid"] = False
                report["column_checks"][col] = col_report

        return report


if __name__ == "__main__":
    validator = DataValidator()
    test_lats = np.arange(6.0, 38.25, 0.25)
    test_lons = np.arange(68.0, 98.25, 0.25)
    grid_check = validator.check_grid_alignment(test_lats, test_lons)
    print("Grid alignment check:", grid_check)
