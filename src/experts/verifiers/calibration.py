"""Calibration verifier for Wai Ultra uncertainty intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.orchestration.protocol import SubtaskKind, VerifierResult, VerifierVerdict


@dataclass
class CalibrationVerifier:
    """Check whether confidence and interval width are justified by support."""

    expert_id: str = "calibration_verifier"

    def verify(self, role_input: Any) -> VerifierResult:
        context = role_input.context
        visible_messages = role_input.visible_messages
        candidate = _latest_candidate(visible_messages)
        if candidate is None:
            return VerifierResult(
                verdict=VerifierVerdict.ABSTAIN,
                problems_found=["no candidate forecast was visible"],
            )
        width = float(candidate["upper_m"]) - float(candidate["lower_m"])
        confidence = float(candidate["confidence"])
        horizon_hours = float(context.horizon_minutes) / 60.0
        residual = abs(float(context.recent_noaa_residual_m or 0.0))
        expected_min_width = 0.08 + 0.02 * horizon_hours + 0.35 * residual
        coverage_estimate = _coverage_estimate(context, candidate)
        problems = []
        if width < expected_min_width:
            problems.append("interval is narrow relative to horizon/residual support")
        if confidence > 0.85 and coverage_estimate < 0.75:
            problems.append("confidence is high relative to historical coverage estimate")
        if width > 2.5:
            problems.append("interval is overly wide for operational use")

        if problems and any("narrow" in problem for problem in problems):
            verdict = VerifierVerdict.REPLAN
            next_subtask = SubtaskKind.ESTIMATE_UNCERTAINTY
            next_expert = "ensemble_synthesis"
        elif problems:
            verdict = VerifierVerdict.CONTINUE
            next_subtask = SubtaskKind.VERIFY_PHYSICS
            next_expert = "physics_datum_verifier"
        else:
            verdict = VerifierVerdict.ACCEPT
            next_subtask = None
            next_expert = None
        return VerifierResult(
            verdict=verdict,
            problems_found=problems,
            confidence_adjustment=-0.08 * len(problems),
            interval_adjustment_recommendation=max(1.0, expected_min_width / max(width, 1e-6)),
            requested_evidence=["wider_or_calibrated_interval"] if problems else [],
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


def _coverage_estimate(context: Any, candidate: dict[str, Any]) -> float:
    experts = candidate.get("experts_used", [])
    if not experts:
        return 0.5
    values = []
    for expert in experts:
        if expert in context.recent_model_performance:
            values.append(float(context.recent_model_performance[expert]))
    if not values:
        return 0.7
    return max(0.0, min(1.0, sum(values) / len(values)))
