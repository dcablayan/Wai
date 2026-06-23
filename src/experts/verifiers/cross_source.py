"""Cross-source verifier for Wai Ultra candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.orchestration.protocol import SubtaskKind, VerifierResult, VerifierVerdict


@dataclass
class CrossSourceVerifier:
    """Check local, NOAA, regional, and baseline consistency."""

    expert_id: str = "cross_source_verifier"
    high_disagreement_m: float = 0.4

    def verify(self, context: Any, visible_messages: list[Any]) -> VerifierResult:
        candidate = _latest_candidate(visible_messages)
        if candidate is None:
            return VerifierResult(
                verdict=VerifierVerdict.ABSTAIN,
                problems_found=["no candidate forecast was visible"],
                requested_evidence=["candidate_forecast"],
            )
        problems: list[str] = []
        requested: list[str] = []
        expert_values = _visible_worker_values(visible_messages)
        if len(expert_values) >= 2:
            disagreement = float(np.max(list(expert_values.values())) - np.min(list(expert_values.values())))
            if disagreement > self.high_disagreement_m:
                problems.append(f"visible worker disagreement is high ({disagreement:.2f} m)")
                requested.append("synthesis_from_allowed_workers")
        elif len(expert_values) == 1 and abs(float(context.recent_noaa_residual_m or 0.0)) >= 0.25:
            problems.append("event-like NOAA residual has only one visible forecast source")
            requested.append("independent_regional_or_transfer_worker")

        baseline = context.noaa_tide_prediction or context.local_tide_prediction
        if baseline is not None:
            departure = abs(float(candidate["forecast_m"]) - float(baseline["water_level_m"]))
            if departure > 0.75 and abs(float(context.recent_noaa_residual_m or 0.0)) < 0.25:
                problems.append("candidate departs strongly from tide baseline without matching NOAA residual support")
                requested.append("physics_or_fallback_check")

        if problems and "synthesis_from_allowed_workers" in requested and len(expert_values) >= 2:
            verdict = VerifierVerdict.REPLAN
            next_subtask = SubtaskKind.SYNTHESIZE_FORECASTS
            next_expert = "ensemble_synthesis"
        elif problems:
            verdict = VerifierVerdict.CONTINUE
            next_subtask = SubtaskKind.TRANSFER_REGIONAL_SIGNAL
            next_expert = "regional_to_local_residual"
        else:
            verdict = VerifierVerdict.ACCEPT
            next_subtask = None
            next_expert = None

        return VerifierResult(
            verdict=verdict,
            problems_found=problems,
            confidence_adjustment=-0.12 if problems else 0.0,
            interval_adjustment_recommendation=1.4 if problems else 1.0,
            requested_evidence=requested,
            recommended_next_subtask=next_subtask,
            recommended_next_expert_or_verifier=next_expert,
            safe_fallback_required=False,
        )


def _latest_candidate(visible_messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(visible_messages):
        candidate = message.structured_result.get("forecast")
        if candidate:
            return candidate
    return None


def _visible_worker_values(visible_messages: list[Any]) -> dict[str, float]:
    values = {}
    for message in visible_messages:
        forecast = message.structured_result.get("forecast")
        if forecast and message.role.value == "WORKER":
            values[message.expert_id] = float(forecast["forecast_m"])
    return values
