"""Numerical synthesis worker for allowed worker forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.experts.base import ExpertForecast
from src.orchestration.combiner import ForecastCombiner


@dataclass
class EnsembleSynthesisWorker:
    """Synthesize only the worker forecasts explicitly exposed by access edges."""

    expert_id: str = "ensemble_synthesis"
    method: str = "weighted_median"

    def run(self, context: Any, visible_messages: list[Any]) -> dict[str, Any]:
        forecasts = []
        used_turns = []
        for message in visible_messages:
            forecast_payload = message.structured_result.get("forecast")
            if not forecast_payload:
                continue
            forecasts.append(_forecast_from_payload(context, forecast_payload, message.expert_id))
            used_turns.append(message.turn_id)
        if not forecasts:
            return {
                "forecast": None,
                "worker_status": "unavailable",
                "message": "synthesis received no successful allowed worker outputs",
                "allowed_input_turns": [message.turn_id for message in visible_messages],
            }

        combined = ForecastCombiner().combine(forecasts, method=self.method)
        values = np.array([float(f.predicted_water_level_m) for f in forecasts], dtype=float)
        disagreement = float(np.max(values) - np.min(values)) if len(values) > 1 else 0.0
        confidence = max(0.0, min(1.0, combined.confidence - min(0.25, disagreement * 0.2)))
        half_width = max(
            combined.forecast_m - combined.lower_m,
            combined.upper_m - combined.forecast_m,
            0.04 + 0.5 * disagreement,
        )
        lower = float(combined.forecast_m - half_width)
        upper = float(combined.forecast_m + half_width)

        payload = {
            "forecast_m": float(combined.forecast_m),
            "lower_m": lower,
            "upper_m": upper,
            "confidence": confidence,
            "experts_used": list(combined.experts_used),
            "method": f"ensemble_{self.method}",
            "diagnostics": {
                "contributing_expert_weights": combined.diagnostics.get("weights", {}),
                "disagreement_m": disagreement,
                "allowed_input_turns": used_turns,
                "assumptions": [
                    "only forecasts in the access list were visible to synthesis",
                    "interval is widened when allowed workers disagree",
                ],
            },
        }
        return {
            "forecast": payload,
            "worker_status": "success",
            "message": "",
            "allowed_input_turns": used_turns,
            "contributing_expert_weights": combined.diagnostics.get("weights", {}),
            "disagreement_diagnostics": {
                "disagreement_m": disagreement,
                "n_allowed_forecasts": len(forecasts),
            },
            "assumptions": payload["diagnostics"]["assumptions"],
        }


def _forecast_from_payload(context: Any, payload: dict[str, Any], model_name: str) -> ExpertForecast:
    return ExpertForecast(
        model_name=model_name,
        forecast_time_utc=context.forecast_time_utc,
        target_time_utc=context.target_time_utc,
        horizon_minutes=context.horizon_minutes,
        predicted_water_level_m=float(payload["forecast_m"]),
        lower_m=float(payload["lower_m"]),
        upper_m=float(payload["upper_m"]),
        confidence=float(payload["confidence"]),
        diagnostics=dict(payload.get("diagnostics", {})),
    )
