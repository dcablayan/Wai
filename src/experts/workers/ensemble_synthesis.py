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

    allow_safe_fallback: bool = False

    def run(self, role_input: Any) -> dict[str, Any]:
        context = role_input.context
        visible_messages = role_input.visible_messages
        forecasts = []
        used_turns = []
        leaf_experts: list[str] = []
        reused_any = False
        for message in visible_messages:
            if message.expert_id == self.expert_id:
                continue
            forecast_payload = message.structured_result.get("forecast")
            if not forecast_payload:
                continue
            leaves = list(forecast_payload.get("leaf_experts", forecast_payload.get("experts_used", [message.expert_id])))
            if not self.allow_safe_fallback and "safe_fallback" in leaves:
                continue
            if any(leaf in leaf_experts for leaf in leaves):
                continue
            forecasts.append(_forecast_from_payload(context, forecast_payload, message.expert_id))
            used_turns.append(message.turn_id)
            leaf_experts.extend(leaves)
            reused_any = reused_any or bool(message.structured_result.get("reused", False))
        unique_leaf_experts = sorted(set(leaf_experts))
        if len(unique_leaf_experts) < 2:
            return {
                "forecast": None,
                "worker_status": "unavailable",
                "message": "synthesis requires at least two distinct successful base numerical experts",
                "allowed_input_turns": [message.turn_id for message in visible_messages],
                "leaf_experts": unique_leaf_experts,
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
            "leaf_experts": unique_leaf_experts,
            "input_turn_ids": used_turns,
            "method": f"ensemble_{self.method}",
            "diagnostics": {
                "contributing_expert_weights": combined.diagnostics.get("weights", {}),
                "disagreement_m": disagreement,
                "allowed_input_turns": used_turns,
                "leaf_experts": unique_leaf_experts,
                "any_contributor_reused": reused_any,
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
            "leaf_experts": unique_leaf_experts,
            "input_turn_ids": used_turns,
            "any_contributor_reused": reused_any,
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
