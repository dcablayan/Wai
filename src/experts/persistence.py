"""Local persistence expert."""

from __future__ import annotations

from src.experts.base import ExpertForecast, ForecastExpert, clamp_confidence, interval
from src.experts.capabilities import LATENCY_INSTANT, ExpertSpec


class LocalPersistenceExpert(ForecastExpert):
    """Short-horizon local forecast from latest Hohonu level and trend."""

    model_name = "local_persistence"
    spec = ExpertSpec(
        model_name="local_persistence",
        required_sources=("hohonu_observation",),
        requires_local_obs=True,
        max_horizon_minutes=12 * 60,
        latency_class=LATENCY_INSTANT,
        compute_cost=1.0,
        notes="Latest local level plus recent trend; degrades past ~12h.",
    )

    def forecast(self, context) -> ExpertForecast:
        obs = context.latest_hohonu_observation
        if not obs:
            return self.unavailable(context, "latest Hohonu observation is missing")
        if not context.hohonu_qc_ok:
            return self.unavailable(context, "latest Hohonu observation failed QC")
        freshness = context.observation_freshness_seconds.get("hohonu", float("inf"))
        if freshness > 3 * 60 * 60:
            return self.unavailable(context, "latest Hohonu observation is stale")

        trend = context.recent_hohonu_trend_m_per_hour or 0.0
        horizon_hours = context.horizon_minutes / 60.0
        prediction = float(obs["water_level_m"] + trend * horizon_hours)
        half_width = 0.04 + 0.02 * horizon_hours + min(abs(trend) * horizon_hours, 0.25)
        lower, upper = interval(prediction, half_width)
        confidence = clamp_confidence(0.9 - freshness / (6 * 60 * 60) - 0.02 * horizon_hours)
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
                "latest_observation_m": float(obs["water_level_m"]),
                "trend_m_per_hour": float(trend),
                "freshness_seconds": float(freshness),
            },
        )
