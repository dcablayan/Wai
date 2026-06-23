"""Physical and datum verifier for Wai Ultra candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.orchestration.protocol import SubtaskKind, VerifierResult, VerifierVerdict


@dataclass
class PhysicsAndDatumVerifier:
    """Check basic physics, datum compatibility, source freshness, and intervals."""

    expert_id: str = "physics_datum_verifier"
    plausible_min_m: float = -5.0
    plausible_max_m: float = 5.0
    suspicious_jump_m: float = 1.5

    def verify(self, role_input: Any) -> VerifierResult:
        context = role_input.context
        visible_messages = role_input.visible_messages
        candidate = _latest_candidate(visible_messages)
        if candidate is None:
            return VerifierResult(
                verdict=VerifierVerdict.ABSTAIN,
                problems_found=["no candidate forecast was visible"],
                requested_evidence=["candidate_forecast"],
                recommended_next_subtask=SubtaskKind.FORECAST_LOCAL_LEVEL,
            )
        problems: list[str] = []
        forecast = float(candidate["forecast_m"])
        lower = float(candidate["lower_m"])
        upper = float(candidate["upper_m"])
        if lower > upper:
            problems.append("invalid interval: lower bound exceeds upper bound")
        if not (lower <= forecast <= upper):
            problems.append("point forecast is outside interval")
        if not (self.plausible_min_m <= forecast <= self.plausible_max_m):
            problems.append("forecast is outside configured plausible water-level range")
        latest = context.latest_hohonu_observation
        if latest is not None:
            jump = abs(forecast - float(latest["water_level_m"]))
            if jump > self.suspicious_jump_m:
                problems.append(f"physically suspicious jump from latest local observation ({jump:.2f} m)")
        if context.datum is None:
            problems.append("datum is unknown")
        if not context.hohonu_qc_ok and "local_persistence" in candidate.get("experts_used", []):
            problems.append("candidate used local persistence after Hohonu QC failure")
        if not context.noaa_qc_ok and any(
            expert in candidate.get("experts_used", [])
            for expert in ("noaa_residual", "regional_to_local_residual")
        ):
            problems.append("candidate used NOAA residual path after NOAA QC failure")

        if any("outside configured plausible" in p or "invalid interval" in p for p in problems):
            verdict = VerifierVerdict.REJECT
        elif problems:
            verdict = VerifierVerdict.REPLAN
        else:
            verdict = VerifierVerdict.ACCEPT
        return VerifierResult(
            verdict=verdict,
            problems_found=problems,
            confidence_adjustment=-0.1 * len(problems),
            interval_adjustment_recommendation=1.5 if problems else 1.0,
            requested_evidence=["safe_fallback"] if verdict is VerifierVerdict.REJECT else [],
            recommended_next_subtask=SubtaskKind.FORECAST_LOCAL_LEVEL if verdict is not VerifierVerdict.ACCEPT else None,
            recommended_next_expert_or_verifier="safe_fallback" if verdict is VerifierVerdict.REJECT else None,
            safe_fallback_required=verdict is VerifierVerdict.REJECT,
        )


def _latest_candidate(visible_messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(visible_messages):
        candidate = message.structured_result.get("forecast")
        if candidate:
            return candidate
    return None
