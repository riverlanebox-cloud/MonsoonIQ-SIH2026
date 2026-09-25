"""
MonsoonIQ Geographic & District Aggregation Utilities.
Provides high-performance spatial raster-to-polygon area-weighted mapping
between 0.25° grid cells and Indian district boundaries.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from shapely.geometry import shape, Point, Polygon

logger = logging.getLogger(__name__)


class DistrictAggregator:
    """Computes area-weighted district aggregations from 0.25° gridded arrays."""

    def __init__(self, geojson_path: str = "data/geojson/india_districts.geojson",
                 lats: np.ndarray = None, lons: np.ndarray = None):
        self.geojson_path = geojson_path
        self.districts = self._load_districts()
        self.lats = lats
        self.lons = lons
        self.weights_cache = {}
        if lats is not None and lons is not None:
            self._precompute_district_grid_indices(lats, lons)

    def _load_districts(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.geojson_path):
            raise FileNotFoundError(f"District GeoJSON file not found: {self.geojson_path}")
        with open(self.geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        districts = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            geom = shape(feat.get("geometry"))
            districts.append({
                "district_id": props.get("district_id", props.get("id")),
                "district_name": props.get("district_name", props.get("name")),
                "state_name": props.get("state_name", props.get("state")),
                "zone": props.get("zone", "Central India"),
                "centroid_lat": props.get("centroid_lat", geom.centroid.y),
                "centroid_lon": props.get("centroid_lon", geom.centroid.x),
                "elevation_m": props.get("elevation_m", 100),
                "geometry": geom
            })
        logger.info(f"Loaded {len(districts)} districts from {self.geojson_path}")
        return districts

    def _precompute_district_grid_indices(self, lats: np.ndarray, lons: np.ndarray):
        """Map each district to intersecting grid points and their relative area weights."""
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        cell_size = 0.25

        for dist in self.districts:
            did = dist["district_id"]
            geom = dist["geometry"]
            minx, miny, maxx, maxy = geom.bounds

            # Filter candidates
            lat_mask = (lats >= miny - cell_size) & (lats <= maxy + cell_size)
            lon_mask = (lons >= minx - cell_size) & (lons <= maxx + cell_size)

            sub_lats = np.where(lat_mask)[0]
            sub_lons = np.where(lon_mask)[0]

            indices = []
            weights = []

            for i in sub_lats:
                lat = lats[i]
                for j in sub_lons:
                    lon = lons[j]
                    cell_poly = Polygon([
                        (lon - cell_size/2, lat - cell_size/2),
                        (lon + cell_size/2, lat - cell_size/2),
                        (lon + cell_size/2, lat + cell_size/2),
                        (lon - cell_size/2, lat + cell_size/2),
                    ])
                    if geom.intersects(cell_poly):
                        intersection_area = geom.intersection(cell_poly).area
                        if intersection_area > 1e-6:
                            indices.append((i, j))
                            weights.append(intersection_area)

            # Fallback to nearest grid cell if polygon is small
            if len(indices) == 0:
                c_lat = dist["centroid_lat"]
                c_lon = dist["centroid_lon"]
                nearest_i = int(np.argmin(np.abs(lats - c_lat)))
                nearest_j = int(np.argmin(np.abs(lons - c_lon)))
                indices = [(nearest_i, nearest_j)]
                weights = [1.0]

            weights = np.array(weights, dtype=np.float32)
            total_w = weights.sum()
            if total_w > 0:
                weights /= total_w
            else:
                weights = np.ones_like(weights) / len(weights)

            self.weights_cache[did] = {
                "indices": indices,
                "weights": weights
            }

        logger.info(f"Precomputed spatial weights for {len(self.weights_cache)} districts.")

    def aggregate_grid_to_districts(self, grid_array_2d: np.ndarray,
                                   thresholds: List[float] = [64.5, 115.6, 204.5]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate 2D grid (n_lat, n_lon) to district statistics:
        - area-weighted mean
        - area-weighted max
        - fraction exceeding each threshold
        """
        results = {}
        for dist in self.districts:
            did = dist["district_id"]
            mapping = self.weights_cache.get(did)
            if not mapping:
                continue

            indices = mapping["indices"]
            weights = mapping["weights"]

            vals = np.array([grid_array_2d[i, j] for (i, j) in indices], dtype=np.float32)

            mean_val = float(np.sum(vals * weights))
            max_val = float(np.max(vals))

            exceedance = {}
            for t in thresholds:
                # fraction of district area exceeding threshold t
                frac = float(np.sum(weights[vals >= t]))
                exceedance[f"frac_gt_{t:.1f}"] = round(frac, 4)

            results[did] = {
                "mean": round(mean_val, 2),
                "max": round(max_val, 2),
                "p90": round(float(np.percentile(vals, 90)), 2),
                **exceedance
            }
        return results
