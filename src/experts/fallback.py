"""Safe baseline forecast expert."""

from __future__ import annotations

from src.experts.base import ExpertForecast, ForecastExpert, clamp_confidence, interval
from src.experts.capabilities import LATENCY_INSTANT, ExpertSpec


class SafeFallbackExpert(ForecastExpert):
    """Return a conservative tide-only baseline when live observations fail."""

    model_name = "safe_fallback"
    spec = ExpertSpec(
        model_name="safe_fallback",
        required_sources=("tide_prediction",),
        requires_tide=True,
        is_safe_baseline=True,
        latency_class=LATENCY_INSTANT,
        compute_cost=0.5,
        notes="Conservative tide-only baseline; the always-available safety net.",
    )

    def forecast(self, context) -> ExpertForecast:
        tide = context.local_tide_prediction or context.noaa_tide_prediction
        if tide is None:
            return self.unavailable(context, "safe baseline unavailable: no tide prediction")
        prediction = float(tide["water_level_m"])
        horizon_hours = context.horizon_minutes / 60.0
        half_width = 0.15 + 0.025 * min(horizon_hours, 24.0)
        lower, upper = interval(prediction, half_width)
        return ExpertForecast(
            model_name=self.model_name,
            forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc,
            horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=prediction,
            lower_m=lower,
            upper_m=upper,
            confidence=clamp_confidence(0.55 - 0.004 * horizon_hours),
            diagnostics={
                "fallback_reason": "tide-only baseline",
                "tide_source": tide.get("source"),
            },
        )
