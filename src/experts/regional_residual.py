"""Regional-to-local residual transfer expert."""

from __future__ import annotations

import pandas as pd

from src.experts.base import ExpertForecast, ForecastExpert, clamp_confidence, interval
from src.experts.capabilities import LATENCY_FAST, ExpertSpec


class RegionalToLocalResidualExpert(ForecastExpert):
    """Transfer NOAA residuals to a paired local station with scale and lag.

    The paired NOAA station's non-tidal residual is assumed to reach the local
    station ``lag_minutes`` later.  To predict the local residual at the target
    time we therefore want the NOAA residual observed at ``target_time -
    lag_minutes``.  When that lagged source time is at or before the forecast
    origin we use the *actual observed* lagged residual (a genuine application of
    the lag); when it would require a future NOAA observation we fall back to a
    clearly documented persistence assumption (the most recent observed
    residual) rather than inventing future data.
    """

    model_name = "regional_to_local_residual"
    spec = ExpertSpec(
        model_name="regional_to_local_residual",
        required_sources=("noaa_observation", "tide_prediction"),
        requires_noaa_obs=True,
        requires_tide=True,
        max_horizon_minutes=48 * 60,
        latency_class=LATENCY_FAST,
        compute_cost=2.0,
        notes="Scales and lags the paired NOAA residual onto the local station.",
    )

    def forecast(self, context) -> ExpertForecast:
        tide = context.local_tide_prediction or context.noaa_tide_prediction
        if tide is None:
            return self.unavailable(context, "no local or NOAA tide prediction is available")
        if context.recent_noaa_residual_m is None:
            return self.unavailable(context, "recent NOAA residual is missing")
        if not context.noaa_qc_ok:
            return self.unavailable(context, "latest NOAA observation failed QC")

        lag_minutes = int(context.station_pair.lag_minutes)
        source_time = context.target_time_utc - pd.Timedelta(minutes=lag_minutes)
        lag_applied = source_time <= context.forecast_time_utc
        if not lag_applied:
            # Lagged source residual is in the future; persist the latest one.
            source_time = context.forecast_time_utc

        source_residual = context.noaa_residual_at(source_time)
        if source_residual is None:
            source_residual = float(context.recent_noaa_residual_m)

        scale = float(context.station_pair.residual_scale)
        residual = scale * float(source_residual)
        prediction = float(tide["water_level_m"] + residual)
        horizon_hours = context.horizon_minutes / 60.0
        # Persistence-assumed lag is less trustworthy than a genuine lagged lookup.
        lag_penalty = 0.0 if (lag_applied or lag_minutes == 0) else 0.05
        half_width = 0.1 + 0.025 * horizon_hours + 0.35 * abs(residual) + (0.05 if not lag_applied and lag_minutes else 0.0)
        lower, upper = interval(prediction, half_width)
        confidence = clamp_confidence(
            0.72 - 0.008 * horizon_hours - abs(scale - 1.0) * 0.1 - lag_penalty
        )
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
                "lag_minutes": lag_minutes,
                "lag_source_time_utc": str(source_time),
                "lag_applied": bool(lag_applied or lag_minutes == 0),
                "source_residual_m": float(source_residual),
                "transferred_residual_m": residual,
            },
        )
