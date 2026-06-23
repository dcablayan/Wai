"""Local tide or harmonic-baseline expert."""

from __future__ import annotations

from src.experts.base import ExpertForecast, ForecastExpert, clamp_confidence, interval
from src.experts.capabilities import LATENCY_INSTANT, ExpertSpec


class LocalTideExpert(ForecastExpert):
    """Use the best available local tide prediction for the target time."""

    model_name = "local_tide"
    spec = ExpertSpec(
        model_name="local_tide",
        required_sources=("tide_prediction",),
        requires_tide=True,
        latency_class=LATENCY_INSTANT,
        compute_cost=1.0,
        notes="Deterministic tide schedule lookup; valid at all horizons.",
    )

    def forecast(self, context) -> ExpertForecast:
        tide = context.local_tide_prediction or context.noaa_tide_prediction
        if not tide:
            return self.unavailable(context, "no tide prediction is available")

        prediction = float(tide["water_level_m"])
        horizon_hours = context.horizon_minutes / 60.0
        half_width = 0.06 + 0.01 * min(horizon_hours, 24.0)
        lower, upper = interval(prediction, half_width)
        confidence = clamp_confidence(0.78 - 0.005 * horizon_hours)
        return ExpertForecast(
            model_name=self.model_name,
            forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc,
            horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=prediction,
            lower_m=lower,
            upper_m=upper,
            confidence=confidence,
            diagnostics={
                "tide_source": tide.get("source"),
                "tide_timestamp_utc": str(tide.get("timestamp_utc")),
                "tide_phase": context.tide_phase,
            },
        )
