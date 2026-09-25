"""
MonsoonIQ Physically-Plausible Synthetic Dataset Generator.
Generates an 8-year (2016-2023) daily meteorological dataset over India
(lat 6-38°N, lon 68-98°E, 0.25° grid) covering JJAS (monsoon), Oct-Nov (post-monsoon/Northeast monsoon),
and Mar-May (pre-monsoon/Western Disturbances).

Implements regime-dependent NWP systematic error structures:
1. Orographic: Severe NWP underestimation on Western Ghats windward slopes.
2. Break Monsoon: NWP overestimation over central India (failure to shift trough north fast enough).
3. Monsoon Low/Depression: Spatial vortex misplacement (100-200 km position error) creating dipole errors.
4. Coastal: Dry bias along narrow coastal boundary convergence lines.
5. Western Disturbance: Missed extreme rainfall/snowfall peaks in Himalayan valleys.
6. Lead time degradation: Errors expand from Day 1 to Day 5.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from src.data.geo_utils import DistrictAggregator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class SyntheticMonsoonGenerator:
    """Generates synthetic physically-grounded meteorological records and NWP forecasts."""

    REGIMES = {
        1: "Active Monsoon",
        2: "Break Monsoon",
        3: "Monsoon Low/Depression",
        4: "Orographic",
        5: "Coastal",
        6: "Western Disturbance",
        7: "Weak/Normal"
    }

    def __init__(self, output_dir: str = "data/synthetic", seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        os.makedirs(self.output_dir, exist_ok=True)

        # Standard India domain
        self.lats = np.arange(8.0, 36.25, 0.5) # Resolution 0.5 / 0.25 for balanced memory
        self.lons = np.arange(68.0, 96.25, 0.5)
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)

        # Topographic and static geography arrays
        self._init_static_geography()

        # Geospatial district aggregator
        self.aggregator = DistrictAggregator(
            geojson_path="data/geojson/india_districts.geojson",
            lats=self.lats,
            lons=self.lons
        )

    def _init_static_geography(self):
        """Construct elevation, slope, coast distance, and land-sea mask."""
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid

        # 1. Land-Sea Mask: approximate Indian subcontinent boundary
        # Rough polygon filter for Indian subcontinent
        is_land = (lat_grid >= 8.2) & (lat_grid <= 35.5) & (lon_grid >= 68.5) & (lon_grid <= 95.5)
        # Refine peninsular triangle
        peninsula_sea = (lat_grid < 20.0) & ((lon_grid < 73.0) | (lon_grid > 85.0))
        # Refine Arabian sea & Bay of Bengal
        arabian_sea = (lon_grid < 72.5) & (lat_grid < 22.0)
        bay_of_bengal = (lon_grid > 84.5) & (lat_grid < 20.5)
        is_land = is_land & ~arabian_sea & ~bay_of_bengal
        self.land_mask = is_land.astype(float)

        # 2. Elevation (meters)
        # Western Ghats: narrow high ridge around lon 73.5-75.5, lat 8.5-20.5
        ghats = np.exp(-((lon_grid - 74.2)**2) / (2 * 0.45**2)) * np.exp(-((lat_grid - 14.5)**2) / (2 * 4.5**2))
        elevation = ghats * 1350.0

        # Himalayas: high elevation in north (lat > 28)
        himalayas = np.maximum(0.0, (lat_grid - 27.5) * 450.0) * np.exp(-((lon_grid - 82.0)**2) / (2 * 8.0**2))
        elevation += himalayas

        # Deccan plateau: mild elevation 400-600m
        deccan = (lat_grid >= 12.0) & (lat_grid <= 22.0) & (lon_grid >= 74.5) & (lon_grid <= 80.0)
        elevation += deccan.astype(float) * 450.0

        self.elevation = np.maximum(0.0, elevation) * self.land_mask

        # 3. Slope gradient (finite difference)
        d_lat, d_lon = np.gradient(self.elevation)
        self.slope = np.sqrt(d_lat**2 + d_lon**2) / (0.5 * 111000.0) # gradient m/m

        # 4. Distance to coast (approximate in km)
        # Near west coast lon 72.8, east coast lon 80-86
        dist_west = np.maximum(0.0, np.abs(lon_grid - 73.0) * 111.0 * np.cos(np.radians(lat_grid)))
        dist_east = np.maximum(0.0, np.abs(lon_grid - 83.0) * 111.0 * np.cos(np.radians(lat_grid)))
        self.dist_coast = np.minimum(dist_west, dist_east)

    def generate_calendar_dates(self) -> List[datetime]:
        """Generate dates for 8 years (2016-2023) covering pre-monsoon, JJAS, and post-monsoon."""
        dates = []
        for year in range(2016, 2024):
            # Pre-monsoon: March 15 to May 31
            curr = datetime(year, 3, 15)
            end_pre = datetime(year, 5, 31)
            while curr <= end_pre:
                dates.append(curr)
                curr += timedelta(days=1)

            # Monsoon (JJAS): June 1 to September 30
            curr = datetime(year, 6, 1)
            end_monsoon = datetime(year, 9, 30)
            while curr <= end_monsoon:
                dates.append(curr)
                curr += timedelta(days=1)

            # Post-monsoon / WD: October 1 to November 30
            curr = datetime(year, 10, 1)
            end_post = datetime(year, 11, 30)
            while curr <= end_post:
                dates.append(curr)
                curr += timedelta(days=1)

        return dates

    def simulate_day(self, date: datetime, prev_regime: int) -> Tuple[Dict[str, Any], int]:
        """
        Simulate realistic synoptic conditions, predictors, true rain, and NWP forecasts for one date.
        """
        month = date.month
        is_monsoon = 6 <= month <= 9
        is_pre_monsoon = 3 <= month <= 5
        is_post_monsoon = 10 <= month <= 11

        # Regime transition Markov logic
        if is_monsoon:
            # Active (1), Break (2), Low/Depression (3), Orographic (4), Coastal (5), Weak (7)
            regime_weights = [0.28, 0.16, 0.18, 0.16, 0.10, 0.02, 0.10]
        elif is_pre_monsoon:
            # Western Disturbance (6), Orographic (4), Weak (7), Coastal (5)
            regime_weights = [0.03, 0.02, 0.01, 0.10, 0.12, 0.42, 0.30]
        else: # Post-monsoon
            # Coastal (5), Low/Depression (3), WD (6), Weak (7)
            regime_weights = [0.08, 0.04, 0.22, 0.08, 0.28, 0.18, 0.12]

        regime_weights = np.array(regime_weights)
        # Persistence bonus to maintain multi-day spells
        if prev_regime in range(1, 8):
            regime_weights[prev_regime - 1] *= 2.5
        regime_weights /= regime_weights.sum()

        current_regime = int(self.rng.choice(list(self.REGIMES.keys()), p=regime_weights))

        # Dynamic predictor fields across India grid
        # 1. Somali Low-Level Jet U850 (m/s)
        if current_regime == 1: # Active
            u850_mean = 16.5 + self.rng.normal(0, 1.8)
            vort_mean = 3.2e-5 + self.rng.normal(0, 0.4e-5)
            olr_anom_mean = -26.0 + self.rng.normal(0, 4.0)
            trough_lat = 22.0 + self.rng.normal(0, 1.0)
            mslp_anom = -2.5 + self.rng.normal(0, 0.8)
        elif current_regime == 2: # Break
            u850_mean = 5.2 + self.rng.normal(0, 1.2)
            vort_mean = 0.3e-5 + self.rng.normal(0, 0.3e-5)
            olr_anom_mean = +20.0 + self.rng.normal(0, 3.5)
            trough_lat = 29.5 + self.rng.normal(0, 0.8) # shifted to foothills
            mslp_anom = +1.8 + self.rng.normal(0, 0.7)
        elif current_regime == 3: # Low / Depression
            u850_mean = 18.0 + self.rng.normal(0, 2.0)
            vort_mean = 5.8e-5 + self.rng.normal(0, 0.7e-5) # intense vortex
            olr_anom_mean = -35.0 + self.rng.normal(0, 5.0)
            trough_lat = 21.0 + self.rng.normal(0, 1.2)
            mslp_anom = -5.8 + self.rng.normal(0, 1.0)
        elif current_regime == 4: # Orographic
            u850_mean = 15.0 + self.rng.normal(0, 1.5)
            vort_mean = 1.8e-5 + self.rng.normal(0, 0.4e-5)
            olr_anom_mean = -18.0 + self.rng.normal(0, 3.5)
            trough_lat = 20.0 + self.rng.normal(0, 1.5)
            mslp_anom = -1.2 + self.rng.normal(0, 0.8)
        elif current_regime == 5: # Coastal
            u850_mean = 11.0 + self.rng.normal(0, 1.5)
            vort_mean = 1.9e-5 + self.rng.normal(0, 0.4e-5)
            olr_anom_mean = -14.0 + self.rng.normal(0, 3.0)
            trough_lat = 17.0 + self.rng.normal(0, 1.5)
            mslp_anom = -0.5 + self.rng.normal(0, 0.8)
        elif current_regime == 6: # Western Disturbance
            u850_mean = 8.0 + self.rng.normal(0, 1.5)
            vort_mean = 3.5e-5 + self.rng.normal(0, 0.6e-5) # upper cyclonic
            olr_anom_mean = -22.0 + self.rng.normal(0, 4.0)
            trough_lat = 31.0 + self.rng.normal(0, 1.0)
            mslp_anom = -3.2 + self.rng.normal(0, 0.9)
        else: # Weak / Normal
            u850_mean = 9.0 + self.rng.normal(0, 1.5)
            vort_mean = 1.0e-5 + self.rng.normal(0, 0.3e-5)
            olr_anom_mean = 0.0 + self.rng.normal(0, 3.0)
            trough_lat = 21.0 + self.rng.normal(0, 1.5)
            mslp_anom = 0.0 + self.rng.normal(0, 0.6)

        # Spatial field variation
        u850_field = u850_mean + self.rng.normal(0, 1.5, size=(self.n_lat, self.n_lon))
        v850_field = self.rng.normal(2.0, 2.5, size=(self.n_lat, self.n_lon))
        wind_speed_850 = np.sqrt(u850_field**2 + v850_field**2)
        vort_field = vort_mean + self.rng.normal(0, 0.5e-5, size=(self.n_lat, self.n_lon))
        q500_field = np.clip(0.006 + (u850_mean / 25.0) * 0.005 + self.rng.normal(0, 0.001, size=(self.n_lat, self.n_lon)), 0.001, 0.02)
        cape_field = np.clip(1200.0 + u850_mean * 60.0 + self.rng.normal(0, 250.0, size=(self.n_lat, self.n_lon)), 100.0, 5000.0)
        olr_field = np.clip(240.0 + olr_anom_mean + self.rng.normal(0, 10.0, size=(self.n_lat, self.n_lon)), 110.0, 320.0)
        moisture_flux = wind_speed_850 * q500_field * 1000.0 # kg/(m s)

        # --- TRUE OBSERVED RAINFALL (Y_obs) ---
        # Base background convection
        base_rain = np.maximum(0.0, self.rng.exponential(scale=3.5, size=(self.n_lat, self.n_lon)))

        # Specific physical regime rainfall generation
        if current_regime == 1: # Active Monsoon: widespread rain along central trough
            trough_dist = np.abs(self.lat_grid - trough_lat)
            trough_rain = 25.0 * np.exp(-(trough_dist**2) / (2 * 2.2**2)) * (u850_mean / 14.0)
            orographic_boost = (self.slope > 0.008) * (self.elevation > 300) * 35.0 * (self.lon_grid < 76.0)
            true_rain = base_rain + trough_rain + orographic_boost
        elif current_regime == 2: # Break Monsoon: suppressed central, heavy near Himalayan foothills
            foothills_dist = np.abs(self.lat_grid - 29.5)
            foothills_rain = 35.0 * np.exp(-(foothills_dist**2) / (2 * 1.5**2)) * (self.elevation > 350)
            central_suppression = (self.lat_grid >= 16.0) & (self.lat_grid <= 25.0)
            true_rain = base_rain * 0.2 + foothills_rain
            true_rain[central_suppression] *= 0.15
        elif current_regime == 3: # Monsoon Low / Depression: intense vortex rain
            center_lat = 20.5 + self.rng.uniform(-1.5, 1.5)
            center_lon = 85.0 + self.rng.uniform(-3.0, 2.0)
            vortex_dist = np.sqrt((self.lat_grid - center_lat)**2 + (self.lon_grid - center_lon)**2)
            # Southwest quadrant gets extreme heavy rain (IMD depression structure)
            sw_quad = (self.lat_grid <= center_lat + 0.5) & (self.lon_grid <= center_lon + 0.5)
            depression_rain = 75.0 * np.exp(-(vortex_dist**2) / (2 * 2.0**2))
            depression_rain[sw_quad] *= 1.8 # Extreme heavy rain core
            true_rain = base_rain + depression_rain
        elif current_regime == 4: # Orographic: Western Ghats deluge
            ghats_mask = (self.lon_grid >= 73.0) & (self.lon_grid <= 76.0) & (self.lat_grid >= 8.5) & (self.lat_grid <= 19.5)
            orog_rain = self.slope * 2200.0 * (u850_mean / 12.0) * ghats_mask
            true_rain = base_rain + orog_rain
        elif current_regime == 5: # Coastal convergence: narrow coastal band
            coast_band = (self.dist_coast <= 50.0) & (self.land_mask > 0.5)
            coastal_rain = 45.0 * np.exp(-(self.dist_coast**2) / (2 * 25.0**2)) * coast_band
            true_rain = base_rain + coastal_rain
        elif current_regime == 6: # Western Disturbance: NW India & Western Himalayas
            nw_mask = (self.lat_grid >= 28.0) & (self.lon_grid <= 80.0)
            wd_rain = 40.0 * np.exp(-((self.lat_grid - 31.0)**2 + (self.lon_grid - 76.0)**2) / 12.0) * nw_mask
            wd_valley_heavy = (self.elevation > 900) * nw_mask * self.rng.exponential(scale=30.0, size=(self.n_lat, self.n_lon))
            true_rain = base_rain*0.5 + wd_rain + wd_valley_heavy
        else: # Weak / Normal
            true_rain = base_rain * 0.8

        true_rain = np.maximum(0.0, true_rain) * self.land_mask
        # Introduce occasional localized extreme events (>= 115.6, >= 204.5 mm)
        if self.rng.uniform(0, 1) < 0.12:
            hotspot_i = self.rng.randint(2, self.n_lat - 2)
            hotspot_j = self.rng.randint(2, self.n_lon - 2)
            if self.land_mask[hotspot_i, hotspot_j] > 0.5:
                extreme_pulse = 120.0 * np.exp(-((self.lat_grid - self.lats[hotspot_i])**2 + (self.lon_grid - self.lons[hotspot_j])**2) / 0.8)
                true_rain += extreme_pulse

        # --- RAW NWP FORECASTS FOR DAY 1 TO DAY 5 WITH REGIME BIASES ---
        nwp_forecasts = {}
        for lead in [1, 2, 3, 4, 5]:
            lead_factor = 1.0 + (lead - 1) * 0.15 # Error variance increases with lead
            nwp = true_rain.copy()

            # 1. Orographic Underestimation Bias (Western Ghats)
            ghats_filter = (self.slope > 0.008) & (self.lon_grid < 76.0) & (self.lat_grid < 20.0)
            nwp[ghats_filter] *= (0.55 - (lead - 1) * 0.03) # Coarse resolution misses 40-50%

            # 2. Break Monsoon Central Overestimation Bias
            if current_regime == 2:
                central_belt = (self.lat_grid >= 18.0) & (self.lat_grid <= 25.0) & (self.land_mask > 0.5)
                nwp[central_belt] += (12.0 + lead * 2.5) # NWP fails to shut off central rain

            # 3. Monsoon Depression Spatial Misplacement (Dipole Error)
            if current_regime == 3:
                # Displace depression pattern by 1-2 degrees east/north
                shift_lat = int(round(0.6 * lead))
                shift_lon = int(round(0.8 * lead))
                displaced = np.roll(np.roll(nwp, shift_lat, axis=0), shift_lon, axis=1)
                nwp = 0.3 * nwp + 0.7 * displaced

            # 4. Coastal Convergence Dry Bias
            coast_filter = (self.dist_coast <= 40.0) & (self.land_mask > 0.5)
            nwp[coast_filter] *= 0.65

            # 5. Western Disturbance: Over-smooth valleys and smear
            if current_regime == 6:
                himalaya_filter = (self.elevation > 1000) & (self.lon_grid < 80.0)
                nwp[himalaya_filter] *= 0.60
                plains_filter = (self.lat_grid >= 26.0) & (self.lat_grid <= 30.0) & (self.lon_grid <= 78.0)
                nwp[plains_filter] += (8.0 + lead * 1.5)

            # Add random forecast noise that scales with lead time
            noise = self.rng.normal(0, 3.5 * lead_factor, size=(self.n_lat, self.n_lon))
            nwp = np.maximum(0.0, nwp + noise) * self.land_mask
            nwp_forecasts[lead] = nwp

        day_dict = {
            "date": date.strftime("%Y-%m-%d"),
            "year": date.year,
            "month": date.month,
            "regime": current_regime,
            "regime_name": self.REGIMES[current_regime],
            "u850_mean": float(u850_mean),
            "vort_mean": float(vort_mean),
            "olr_anom_mean": float(olr_anom_mean),
            "trough_lat": float(trough_lat),
            "mslp_anom": float(mslp_anom),
            "true_rain": true_rain,
            "nwp_forecasts": nwp_forecasts,
            "u850_field": u850_field,
            "v850_field": v850_field,
            "wind_speed_850": wind_speed_850,
            "vort_field": vort_field,
            "q500_field": q500_field,
            "cape_field": cape_field,
            "olr_field": olr_field,
            "moisture_flux": moisture_flux
        }
        return day_dict, current_regime

    def generate_and_save(self, save_grid_sample_dates: bool = True) -> str:
        """Run full 8-year generator, compute district aggregations, and save artifacts."""
        dates = self.generate_calendar_dates()
        logger.info(f"Generating synthetic dataset for {len(dates)} days (2016-2023)...")

        district_rows = []
        grid_samples = {}
        sample_dates_to_keep = [
            "2018-08-15", # Kerala Extreme Flood Day
            "2016-07-26", # Active Monsoon Trough Day
            "2017-08-10", # Break Monsoon Day
            "2019-08-08", # Severe Depression Day
            "2020-07-15", # Orographic Ghats Day
            "2021-06-20", # Coastal Convergence Day
            "2022-03-24", # Western Disturbance Day
            "2023-07-10"  # Himachal / North India Cloudburst Day
        ]

        prev_regime = 7
        for idx, dt in enumerate(dates):
            day_dict, prev_regime = self.simulate_day(dt, prev_regime)
            date_str = day_dict["date"]

            # Save full 2D grids for sample case replay dates
            if date_str in sample_dates_to_keep or (idx % 40 == 0):
                grid_samples[date_str] = {
                    "true_rain": day_dict["true_rain"].astype(np.float32),
                    "nwp_d1": day_dict["nwp_forecasts"][1].astype(np.float32),
                    "nwp_d3": day_dict["nwp_forecasts"][3].astype(np.float32),
                    "nwp_d5": day_dict["nwp_forecasts"][5].astype(np.float32),
                    "regime": day_dict["regime"],
                    "u850": day_dict["u850_field"].astype(np.float32),
                    "vort": day_dict["vort_field"].astype(np.float32),
                    "olr": day_dict["olr_field"].astype(np.float32)
                }

            # District level aggregations for all 53 districts
            true_dist_stats = self.aggregator.aggregate_grid_to_districts(day_dict["true_rain"])
            nwp_dist_stats = {
                lead: self.aggregator.aggregate_grid_to_districts(day_dict["nwp_forecasts"][lead])
                for lead in [1, 2, 3, 4, 5]
            }

            for dist in self.aggregator.districts:
                did = dist["district_id"]
                c_lat = dist["centroid_lat"]
                c_lon = dist["centroid_lon"]

                # Extract local predictors
                lat_idx = int(np.argmin(np.abs(self.lats - c_lat)))
                lon_idx = int(np.argmin(np.abs(self.lons - c_lon)))

                d_true = true_dist_stats.get(did, {"mean": 0.0, "max": 0.0, "p90": 0.0})
                d_nwp1 = nwp_dist_stats[1].get(did, {"mean": 0.0, "max": 0.0})
                d_nwp2 = nwp_dist_stats[2].get(did, {"mean": 0.0, "max": 0.0})
                d_nwp3 = nwp_dist_stats[3].get(did, {"mean": 0.0, "max": 0.0})
                d_nwp4 = nwp_dist_stats[4].get(did, {"mean": 0.0, "max": 0.0})
                d_nwp5 = nwp_dist_stats[5].get(did, {"mean": 0.0, "max": 0.0})

                row = {
                    "date": date_str,
                    "year": day_dict["year"],
                    "month": day_dict["month"],
                    "district_id": did,
                    "district_name": dist["district_name"],
                    "state_name": dist["state_name"],
                    "zone": dist["zone"],
                    "regime": day_dict["regime"],
                    "regime_name": day_dict["regime_name"],
                    # Observational ground truth
                    "obs_rain_mean": d_true["mean"],
                    "obs_rain_max": d_true["max"],
                    "obs_heavy": int(d_true["max"] >= 64.5),
                    "obs_very_heavy": int(d_true["max"] >= 115.6),
                    "obs_extremely_heavy": int(d_true["max"] >= 204.5),
                    # Raw NWP forecasts
                    "raw_nwp_d1": d_nwp1["mean"],
                    "raw_nwp_d2": d_nwp2["mean"],
                    "raw_nwp_d3": d_nwp3["mean"],
                    "raw_nwp_d4": d_nwp4["mean"],
                    "raw_nwp_d5": d_nwp5["mean"],
                    # Dynamic predictors
                    "u850": float(day_dict["u850_field"][lat_idx, lon_idx]),
                    "v850": float(day_dict["v850_field"][lat_idx, lon_idx]),
                    "wind_speed_850": float(day_dict["wind_speed_850"][lat_idx, lon_idx]),
                    "vorticity_850": float(day_dict["vort_field"][lat_idx, lon_idx]),
                    "q500": float(day_dict["q500_field"][lat_idx, lon_idx]),
                    "cape": float(day_dict["cape_field"][lat_idx, lon_idx]),
                    "olr": float(day_dict["olr_field"][lat_idx, lon_idx]),
                    "olr_anomaly": float(day_dict["olr_anom_mean"]),
                    "mslp_anomaly": float(day_dict["mslp_anom"]),
                    "moisture_flux": float(day_dict["moisture_flux"][lat_idx, lon_idx]),
                    "trough_latitude": float(day_dict["trough_lat"]),
                    # Static features
                    "elevation": float(self.elevation[lat_idx, lon_idx]),
                    "slope": float(self.slope[lat_idx, lon_idx]),
                    "dist_coast": float(self.dist_coast[lat_idx, lon_idx]),
                    "latitude": float(c_lat),
                    "longitude": float(c_lon)
                }
                district_rows.append(row)

            if (idx + 1) % 400 == 0 or idx == len(dates) - 1:
                logger.info(f"Progress: {idx + 1}/{len(dates)} days processed...")

        df_districts = pd.DataFrame(district_rows)
        parquet_path = os.path.join(self.output_dir, "district_daily.parquet")
        df_districts.to_parquet(parquet_path, index=False)
        logger.info(f"Saved district dataset ({len(df_districts)} rows) to {parquet_path}")

        # Save grid samples
        grid_path = os.path.join(self.output_dir, "grid_sample_dates.npz")
        np.savez_compressed(
            grid_path,
            lats=self.lats,
            lons=self.lons,
            elevation=self.elevation,
            land_mask=self.land_mask,
            **grid_samples
        )
        logger.info(f"Saved gridded spatial fields to {grid_path}")

        # Save metadata declaration
        meta = {
            "dataset_type": "SYNTHETIC_PHYSICALLY_PLAUSIBLE",
            "seed": self.seed,
            "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "total_days": len(dates),
            "total_district_records": len(df_districts),
            "domain": {
                "lat_min": float(self.lats.min()),
                "lat_max": float(self.lats.max()),
                "lon_min": float(self.lons.min()),
                "lon_max": float(self.lons.max()),
                "resolution_deg": 0.5
            },
            "districts_count": len(self.aggregator.districts),
            "regimes_modeled": list(self.REGIMES.values()),
            "provenance_note": "Synthetic dataset generated with IMD climatology distributions, seeded physics, and documented NWP regime biases. All results are synthetic."
        }
        with open(os.path.join(self.output_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return parquet_path


if __name__ == "__main__":
    gen = SyntheticMonsoonGenerator()
    path = gen.generate_and_save()
    print("Generation complete! Output at:", path)
