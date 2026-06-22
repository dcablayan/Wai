"""NOAA residual expert."""

from __future__ import annotations

from src.experts.base import ExpertForecast, ForecastExpert, clamp_confidence, interval


class NOAAResidualExpert(ForecastExpert):
    """Add recent NOAA non-tidal residual to the target tide prediction."""

    model_name = "noaa_residual"

    def forecast(self, context) -> ExpertForecast:
        if context.noaa_tide_prediction is None:
            return self.unavailable(context, "NOAA tide prediction is missing")
        if context.recent_noaa_residual_m is None:
            return self.unavailable(context, "recent NOAA residual is missing")
        if not context.noaa_qc_ok:
            return self.unavailable(context, "latest NOAA observation failed QC")
        if not context.noaa_is_fresh:
            return self.unavailable(context, "latest NOAA observation is stale")

        residual = float(context.recent_noaa_residual_m)
        trend = context.noaa_residual_trend_m_per_hour or 0.0
        horizon_hours = context.horizon_minutes / 60.0
        adjusted_residual = residual + trend * min(horizon_hours, 3.0)
        prediction = float(context.noaa_tide_prediction["water_level_m"] + adjusted_residual)
        half_width = 0.08 + 0.02 * horizon_hours + 0.25 * abs(adjusted_residual)
        lower, upper = interval(prediction, half_width)
        freshness = context.observation_freshness_seconds.get("noaa", 0.0)
        confidence = clamp_confidence(0.82 - freshness / (12 * 60 * 60) - 0.01 * horizon_hours)
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
                "recent_noaa_residual_m": residual,
                "residual_trend_m_per_hour": float(trend),
                "adjusted_residual_m": float(adjusted_residual),
            },
        )
