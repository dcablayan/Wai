"""Regional-to-local residual transfer expert."""

from __future__ import annotations

from src.experts.base import ExpertForecast, ForecastExpert, clamp_confidence, interval


class RegionalToLocalResidualExpert(ForecastExpert):
    """Transfer NOAA residuals to a paired local station with scale and lag."""

    model_name = "regional_to_local_residual"

    def forecast(self, context) -> ExpertForecast:
        tide = context.local_tide_prediction or context.noaa_tide_prediction
        if tide is None:
            return self.unavailable(context, "no local or NOAA tide prediction is available")
        if context.recent_noaa_residual_m is None:
            return self.unavailable(context, "recent NOAA residual is missing")
        if not context.noaa_qc_ok:
            return self.unavailable(context, "latest NOAA observation failed QC")

        scale = float(context.station_pair.residual_scale)
        residual = scale * float(context.recent_noaa_residual_m)
        prediction = float(tide["water_level_m"] + residual)
        horizon_hours = context.horizon_minutes / 60.0
        half_width = 0.1 + 0.025 * horizon_hours + 0.35 * abs(residual)
        lower, upper = interval(prediction, half_width)
        confidence = clamp_confidence(0.72 - 0.008 * horizon_hours - abs(scale - 1.0) * 0.1)
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
                "paired_noaa_station_id": context.paired_noaa_station_id,
                "residual_scale": scale,
                "lag_minutes": int(context.station_pair.lag_minutes),
                "transferred_residual_m": residual,
            },
        )
