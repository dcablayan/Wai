"""Forecasting expert implementations."""

from src.experts.base import ExpertForecast, ForecastExpert
from src.experts.capabilities import ExpertSpec
from src.experts.fallback import SafeFallbackExpert
from src.experts.harmonic_fallback import HarmonicFallbackExpert
from src.experts.local_tide import LocalTideExpert
from src.experts.noaa_residual import NOAAResidualExpert
from src.experts.persistence import LocalPersistenceExpert
from src.experts.placeholders import (
    LearnedLocalResidualExpert,
    SpatialNeighboringStationExpert,
    WeatherAwareExpert,
)
from src.experts.regional_residual import RegionalToLocalResidualExpert

__all__ = [
    "ExpertForecast",
    "ExpertSpec",
    "ForecastExpert",
    "HarmonicFallbackExpert",
    "LocalPersistenceExpert",
    "LocalTideExpert",
    "NOAAResidualExpert",
    "RegionalToLocalResidualExpert",
    "SafeFallbackExpert",
    "WeatherAwareExpert",
    "SpatialNeighboringStationExpert",
    "LearnedLocalResidualExpert",
]
