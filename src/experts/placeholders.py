"""Explicit placeholders for future experts.

These classes make planned extension points visible without claiming
operational skill from untrained or unavailable models.
"""

from __future__ import annotations

from src.experts.base import ForecastExpert


class WeatherAwareExpert(ForecastExpert):
    model_name = "weather_aware"

    def forecast(self, context):
        return self.unavailable(context, "weather-aware expert is not implemented yet")


class SpatialNeighboringStationExpert(ForecastExpert):
    model_name = "spatial_neighboring_station"

    def forecast(self, context):
        return self.unavailable(context, "neighboring-station expert is not implemented yet")


class LearnedLocalResidualExpert(ForecastExpert):
    model_name = "learned_local_residual"

    def forecast(self, context):
        return self.unavailable(context, "learned local residual expert is not trained yet")
