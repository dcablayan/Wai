"""Event-risk verifier for abnormal residual regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.orchestration.protocol import SubtaskKind, VerifierResult, VerifierVerdict


@dataclass
class EventRiskVerifier:
    """Apply stricter checks during elevated residuals or rapid changes."""

    expert_id: str = "event_risk_verifier"
    elevated_residual_m: float = 0.25
    rapid_change_m_per_hour: float = 0.12

    def verify(self, context: Any, visible_messages: list[Any]) -> VerifierResult:
        candidate = _latest_candidate(visible_messages)
        if candidate is None:
            return VerifierResult(
                verdict=VerifierVerdict.ABSTAIN,
                problems_found=["no candidate forecast was visible"],
            )
        residual = abs(float(context.recent_noaa_residual_m or 0.0))
        trend = abs(float(context.noaa_residual_trend_m_per_hour or 0.0))
        elevated = residual >= self.elevated_residual_m or trend >= self.rapid_change_m_per_hour
        experts = set(candidate.get("experts_used", []))
        width = float(candidate["upper_m"]) - float(candidate["lower_m"])
        problems = []
        requested = []
        if elevated and not (experts & {"noaa_residual", "regional_to_local_residual"}):
            problems.append("elevated residual lacks regional residual evidence")
            requested.append("regional_residual_worker")
        if elevated and len(experts) < 2 and "safe_fallback" not in experts:
            problems.append("event-like regime has only one non-fallback numerical worker")
            requested.append("independent_worker_or_synthesis")
        if elevated and width < 0.18 + 0.5 * residual:
            problems.append("event-risk interval is too narrow")
            requested.append("calibration_or_synthesis")

        if not elevated:
            verdict = VerifierVerdict.ACCEPT
            next_subtask = None
            next_expert = None
        elif problems and "regional_residual_worker" in requested:
            verdict = VerifierVerdict.REPLAN
            next_subtask = SubtaskKind.TRANSFER_REGIONAL_SIGNAL
            next_expert = "regional_to_local_residual"
        elif problems:
            verdict = VerifierVerdict.REPLAN
            next_subtask = SubtaskKind.SYNTHESIZE_FORECASTS
            next_expert = "ensemble_synthesis"
        else:
            verdict = VerifierVerdict.ACCEPT
            next_subtask = None
            next_expert = None

        return VerifierResult(
            verdict=verdict,
            problems_found=problems,
            confidence_adjustment=-0.15 if problems else 0.0,
            interval_adjustment_recommendation=1.6 if problems else 1.0,
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
