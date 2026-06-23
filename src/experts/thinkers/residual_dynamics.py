"""Numerical residual-dynamics diagnostics for Wai Ultra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ResidualDynamicsThinker:
    """Analyze NOAA residual trend, recent changes, and regional transfer signal."""

    expert_id: str = "residual_dynamics_thinker"

    def analyze(self, context: Any, visible_messages: list[Any] | None = None) -> dict[str, Any]:
        residual = float(context.recent_noaa_residual_m or 0.0)
        trend = float(context.noaa_residual_trend_m_per_hour or 0.0)
        residual_series = _residual_series(context)
        change_point_score = _change_point_score(residual_series)
        persistence_score = _persistence_score(residual_series)
        regional_signal = min(1.0, abs(residual) / 0.35 + abs(trend) / 0.25 + change_point_score) / 3.0
        local_trend = abs(float(context.recent_hohonu_trend_m_per_hour or 0.0))
        local_only_score = max(0.0, min(1.0, local_trend / 0.25 - abs(residual) / 0.5))
        scale = float(context.station_pair.residual_scale)
        lag_minutes = int(context.station_pair.lag_minutes)

        recommended_experts = []
        recommended_subtasks = []
        if regional_signal >= local_only_score:
            recommended_experts.extend(["noaa_residual", "regional_to_local_residual"])
            recommended_subtasks.extend(["FORECAST_REGIONAL_RESIDUAL", "TRANSFER_REGIONAL_SIGNAL"])
        else:
            recommended_experts.extend(["local_persistence", "local_tide"])
            recommended_subtasks.append("FORECAST_LOCAL_LEVEL")

        return {
            "forecast_difficulty": float(min(1.0, 0.25 + regional_signal + 0.3 * change_point_score)),
            "event_risk": float(min(1.0, regional_signal + 0.25 * change_point_score)),
            "recent_noaa_residual_m": residual,
            "residual_trend_m_per_hour": trend,
            "change_point_score": float(change_point_score),
            "persistence_score": float(persistence_score),
            "appears_regional": bool(regional_signal >= local_only_score),
            "appears_local": bool(local_only_score > regional_signal),
            "station_pair_scale": scale,
            "station_pair_lag_minutes": lag_minutes,
            "recommended_next_subtasks": recommended_subtasks,
            "recommended_experts": recommended_experts,
            "recommended_verifier_type": "cross_source_verifier",
        }


def _residual_series(context: Any) -> np.ndarray:
    obs = getattr(context, "recent_noaa_observations", None)
    tide = getattr(context, "noaa_tide_predictions", None)
    if obs is None or tide is None or len(obs) < 2 or len(tide) == 0:
        return np.array([], dtype=float)
    residuals = []
    tide_times = tide["timestamp_utc"]
    tide_values = tide["water_level_m"].astype(float).to_numpy()
    for _, row in obs.iterrows():
        idx = (tide_times - row["timestamp_utc"]).abs().idxmin()
        residuals.append(float(row["water_level_m"]) - float(tide_values[idx]))
    return np.array(residuals, dtype=float)


def _change_point_score(values: np.ndarray) -> float:
    if len(values) < 6:
        return 0.0
    midpoint = len(values) // 2
    before = values[:midpoint]
    after = values[midpoint:]
    spread = max(0.05, float(np.nanstd(values)))
    return float(min(1.0, abs(float(np.nanmean(after) - np.nanmean(before))) / (3.0 * spread)))


def _persistence_score(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    diffs = np.diff(values)
    noise = float(np.nanstd(diffs))
    scale = max(0.05, float(np.nanstd(values)))
    return float(max(0.0, min(1.0, 1.0 - noise / (scale + 1e-6))))
