"""Adapters that expose existing numerical experts as Ultra workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.experts.base import ExpertForecast, ForecastExpert


@dataclass
class ForecastWorkerAdapter:
    """Run a ForecastExpert under the worker role contract."""

    expert: ForecastExpert

    @property
    def expert_id(self) -> str:
        return self.expert.model_name

    def run(self, role_input: Any, visible_messages: list[Any] | None = None) -> dict[str, Any]:
        context = getattr(role_input, "context", role_input)
        if self.expert_id in set(getattr(context, "forced_worker_exceptions", set())):
            raise RuntimeError(f"{self.expert_id} forced exception for randomized episode")
        if self.expert_id in set(getattr(context, "forced_worker_timeouts", set())):
            return {
                "forecast": None,
                "worker_status": "timeout",
                "message": f"{self.expert_id} forced timeout for randomized episode",
                "assumptions": _assumptions_for(self.expert_id),
            }
        forecast = self.expert.forecast(context)
        serialized = forecast_to_payload(forecast) if forecast.ok else None
        if serialized is not None and self.expert_id in set(getattr(context, "forced_invalid_intervals", set())):
            serialized["upper_m"] = float(serialized["lower_m"] - 0.05)
            serialized.setdefault("diagnostics", {})["forced_invalid_interval"] = True
        payload = {
            "forecast": serialized,
            "worker_status": forecast.status,
            "message": forecast.message,
            "assumptions": _assumptions_for(forecast.model_name),
        }
        if not forecast.ok:
            payload["unavailable_reason"] = forecast.message
        return payload


def forecast_to_payload(forecast: ExpertForecast) -> dict[str, Any]:
    """Serialize an ExpertForecast into the candidate-forecast shape."""

    return {
        "forecast_m": float(forecast.predicted_water_level_m),
        "lower_m": float(forecast.lower_m),
        "upper_m": float(forecast.upper_m),
        "confidence": float(forecast.confidence),
        "experts_used": [forecast.model_name],
        "leaf_experts": [forecast.model_name],
        "input_turn_ids": [],
        "method": forecast.model_name,
        "diagnostics": dict(forecast.diagnostics),
    }


def _assumptions_for(expert_id: str) -> list[str]:
    assumptions = {
        "local_persistence": [
            "recent local trend remains representative through the horizon",
            "latest Hohonu datum and QC state are valid",
        ],
        "local_tide": [
            "harmonic/tide baseline dominates non-tidal residual over the horizon",
        ],
        "noaa_residual": [
            "recent NOAA residual and short-term residual trend remain informative",
        ],
        "regional_to_local_residual": [
            "paired-station residual scale and lag are applicable at this origin",
        ],
        "safe_fallback": [
            "only a conservative tide baseline is scientifically supportable",
        ],
    }
    return assumptions.get(expert_id, [])
