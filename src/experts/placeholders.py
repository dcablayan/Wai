"""Explicit placeholders for future experts.

These classes make planned extension points visible without claiming
operational skill from untrained or unavailable models.
"""

from __future__ import annotations

from src.experts.base import ForecastExpert
from src.experts.capabilities import LATENCY_MODERATE, LATENCY_SLOW, ExpertSpec


class WeatherAwareExpert(ForecastExpert):
    model_name = "weather_aware"
    spec = ExpertSpec(
        model_name="weather_aware",
        required_sources=("weather_observation", "tide_prediction"),
        requires_tide=True,
        latency_class=LATENCY_SLOW,
        compute_cost=8.0,
        cacheable=False,
        notes="Placeholder; returns unavailable until implemented.",
    )

    def forecast(self, context):
        return self.unavailable(context, "weather-aware expert is not implemented yet")


class SpatialNeighboringStationExpert(ForecastExpert):
    model_name = "spatial_neighboring_station"
    spec = ExpertSpec(
        model_name="spatial_neighboring_station",
        required_sources=("neighboring_station",),
        latency_class=LATENCY_MODERATE,
        compute_cost=4.0,
        notes="Placeholder; returns unavailable until implemented.",
    )

    def forecast(self, context):
        return self.unavailable(context, "neighboring-station expert is not implemented yet")


class LearnedLocalResidualExpert(ForecastExpert):
    model_name = "learned_local_residual"
    spec = ExpertSpec(
        model_name="learned_local_residual",
        required_sources=("hohonu_observation", "tide_prediction"),
        requires_local_obs=True,
        requires_tide=True,
        latency_class=LATENCY_MODERATE,
        compute_cost=6.0,
        notes="Placeholder; returns unavailable until trained.",
    )

    def forecast(self, context):
        return self.unavailable(context, "learned local residual expert is not trained yet")
