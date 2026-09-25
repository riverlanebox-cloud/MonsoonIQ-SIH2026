"""
MonsoonIQ Pydantic Request & Response Schemas.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    data_mode: str = "synthetic"
    models_loaded: bool = True
    total_districts: int = 53


class RegimeInfo(BaseModel):
    regime_id: int
    regime_name: str
    probability: float


class RegimeResponse(BaseModel):
    date: str
    dominant_regime: str
    dominant_regime_id: int
    soft_probabilities: Dict[str, float]


class DistrictForecastItem(BaseModel):
    district_id: str
    district_name: str
    state_name: str
    zone: str
    centroid_lat: float
    centroid_lon: float
    elevation_m: float
    dominant_regime: str
    raw_nwp: float
    monsooniq_corrected: float
    bias_delta: float
    p10: float
    p50: float
    p90: float
    p_heavy: float
    p_very_heavy: float
    p_extremely_heavy: float
    alert_level: str
    alert_code: str


class CorrectedForecastResponse(BaseModel):
    date: str
    lead_time_days: int
    mode: str # 'district' or 'grid'
    provenance: str = "SYNTHETIC_DATASET"
    total_districts: int
    districts: List[DistrictForecastItem]


class HeavyProbabilityResponse(BaseModel):
    date: str
    lead_time_days: int
    provenance: str = "SYNTHETIC_DATASET"
    thresholds_mm: Dict[str, float] = {
        "heavy": 64.5,
        "very_heavy": 115.6,
        "extremely_heavy": 204.5
    }
    district_probabilities: List[Dict[str, Any]]


class SingleDistrictDetailResponse(BaseModel):
    district_id: str
    district_name: str
    state_name: str
    zone: str
    date: str
    lead_time_days: int
    elevation_m: float
    regimes: Dict[str, float]
    dominant_regime: str
    forecast: Dict[str, float]
    uncertainty_bands: Dict[str, float]
    heavy_probabilities: Dict[str, float]
    advisory: Dict[str, Any]
    cap_alert: Dict[str, Any]


class ExplainabilityResponse(BaseModel):
    district_id: str
    date: str
    predicted_regime: str
    soft_probabilities: Dict[str, float]
    top_feature_attributions: List[Dict[str, Any]]
