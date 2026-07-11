"""End-to-end orchestrated forecasting pipeline.

The default path is now an adaptive cascade (capability gate -> primary expert
-> post-forecast assessment -> early stop or conditional escalation -> skill-aware
combination -> dependency-aware verification -> safe fallback).  The legacy flat
router remains available via ``ForecastPipeline(adaptive=False)`` for
benchmarking and backward-compatibility checks.

``run`` keeps the public ``run(context)`` interface and adds backward-compatible
optional parameters:

    pipeline.run(context, precomputed_forecasts=expert_predictions)

which lets historical replay reuse already-computed expert forecasts instead of
running any expert twice at the same origin.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.experts import (
    ForecastExpert,
    HarmonicFallbackExpert,
    LearnedLocalResidualExpert,
    LocalPersistenceExpert,
    LocalTideExpert,
    NOAAResidualExpert,
    RegionalToLocalResidualExpert,
    SafeFallbackExpert,
    SpatialNeighboringStationExpert,
    WeatherAwareExpert,
)
from src.experts.base import ExpertForecast
from src.orchestration.cascade import AdaptiveCascade, ExecutionBudget
from src.orchestration.combiner import ForecastCombiner
from src.orchestration.coordinator_policy import CoordinatorPolicy
from src.orchestration.protocol import ExecutionBudget as UltraExecutionBudget
from src.orchestration.router import RuleBasedOrchestrator
from src.orchestration.skill_store import SkillStore
from src.orchestration.ultra_conductor import UltraConductor
from src.orchestration.verifier import ForecastVerifier

LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Structured final forecast returned by Wai orchestration."""

    station_id: str
    forecast_time_utc: str
    target_time_utc: str
    horizon_minutes: int
    forecast_m: float | None
    lower_m: float | None
    upper_m: float | None
    confidence: float
    regime: str
    experts_used: list[str]
    experts_excluded: dict[str, str]
    combination_method: str
    fallback_used: bool
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    status: str = "available"
    mode: str = "mini"
    coordinator_policy_source: str = "adaptive_cascade"
    coordinator_artifact_version: str = "none"
    number_of_turns: int = 0
    number_of_unique_experts: int = 0
    role_sequence: list[str] = field(default_factory=list)
    executed_topology: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = "completed"
    logical_actions: int = 0
    physical_expert_calls: int = 0
    reused_expert_outputs: int = 0
    unique_experts: int = 0
    verifier_calls: int = 0
    thinker_calls: int = 0
    worker_calls: int = 0
    fallback_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForecastPipeline:
    """Forecast orchestration with mini, ultra, and legacy modes."""

    def __init__(
        self,
        experts: dict[str, ForecastExpert] | None = None,
        orchestrator: RuleBasedOrchestrator | None = None,
        combiner: ForecastCombiner | None = None,
        verifier: ForecastVerifier | None = None,
        *,
        mode: str | None = None,
        adaptive: bool = True,
        cascade: AdaptiveCascade | None = None,
        skill_store: SkillStore | None = None,
        budget: ExecutionBudget | None = None,
        learned_router: Any | None = None,
        interval_inflation: float = 1.8,
        ultra_policy: CoordinatorPolicy | None = None,
        ultra_budget: UltraExecutionBudget | None = None,
    ) -> None:
        resolved_mode = mode or ("mini" if adaptive else "legacy")
        if resolved_mode not in {"mini", "ultra", "legacy"}:
            raise ValueError("ForecastPipeline mode must be one of: mini, ultra, legacy")
        self.mode = resolved_mode
        self.experts = experts or default_experts(include_placeholders=False)
        self.orchestrator = orchestrator or RuleBasedOrchestrator()
        self.combiner = combiner or ForecastCombiner()
        self.verifier = verifier or ForecastVerifier()
        self.adaptive = self.mode == "mini"
        self.skill_store = skill_store or SkillStore()
        self.cascade = cascade or AdaptiveCascade(skill_store=self.skill_store)
        self.budget = budget or ExecutionBudget()
        self.learned_router = learned_router
        self.interval_inflation = interval_inflation
        self.ultra_policy = ultra_policy
        self.ultra_budget = ultra_budget

    # ------------------------------------------------------------------ #
    def run(
        self,
        context,
        *,
        precomputed_forecasts: dict[str, ExpertForecast] | None = None,
        budget: ExecutionBudget | None = None,
        context_build_ms: float = 0.0,
    ) -> ForecastResult:
        if self.mode == "ultra":
            return self._run_ultra(context)
        if self.mode == "legacy" or not self.adaptive:
            return self._run_legacy(context, precomputed_forecasts)
        return self._run_adaptive(
            context,
            precomputed_forecasts=precomputed_forecasts,
            budget=budget or self.budget,
            context_build_ms=context_build_ms,
        )

    # ------------------------------------------------------------------ #
    def _run_adaptive(
        self,
        context,
        *,
        precomputed_forecasts: dict[str, ExpertForecast] | None,
        budget: ExecutionBudget,
        context_build_ms: float,
    ) -> ForecastResult:
        t_total = time.perf_counter()
        results, trace = self.cascade.execute(
            context, self.experts, budget=budget, precomputed=precomputed_forecasts
        )
        regime = trace.regime
        excluded = dict(trace.plan.excluded) if trace.plan else {}
        warnings: list[str] = []
        fallback_used = False
        fallback_reason: str | None = None

        successful = [r.forecast for r in results if r.ok]
        for r in results:
            if r.forecast is not None and not r.ok:
                excluded.setdefault(
                    r.name, r.forecast.message or "expert did not return a successful forecast"
                )
            if r.timed_out:
                excluded.setdefault(r.name, "expert timed out")

        # Shadow-mode learned router (records, never controls the route).
        shadow = self._learned_shadow(context, trace)

        # No successful expert -> reserved safe fallback call.
        if not successful:
            fb = self._safe_fallback(context)
            if fb is not None and fb.ok:
                successful = [fb]
                fallback_used = True
                fallback_reason = "no successful expert forecast; used safe baseline"

        if not successful:
            warnings.append("No valid numerical forecast path is available")
            return self._unavailable_result(
                context, regime, excluded, "none", fallback_used, warnings,
                results, trace, context_build_ms, t_total, fallback_reason, shadow,
            )

        if any(f.model_name == "safe_fallback" for f in successful):
            fallback_used = True
            fallback_reason = fallback_reason or "safe baseline included in forecast set"

        method = _combination_method(successful)
        weights = self._skill_weights(context, successful, regime)
        min_half = self._skill_interval_floor(context, successful, regime)
        t_comb = time.perf_counter()
        try:
            combined = self.combiner.combine(
                successful, method=method, weights=weights,
                horizon_minutes=context.horizon_minutes,
                min_half_width_m=min_half,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Combiner failed: {exc}")
            return self._unavailable_result(
                context, regime, excluded, method, fallback_used, warnings,
                results, trace, context_build_ms, t_total, fallback_reason, shadow,
            )
        combination_ms = (time.perf_counter() - t_comb) * 1000.0

        t_ver = time.perf_counter()
        verified, verification = self.verifier.verify(combined, context=context, forecasts=successful)
        warnings.extend(verification.warnings)

        if verified is None:
            # Post-verifier safe-fallback attempt (once), if recoverable.
            if verification.recoverable and not fallback_used:
                fb = self._safe_fallback(context)
                if fb is not None and fb.ok:
                    fb_combined = self.combiner.combine([fb], horizon_minutes=context.horizon_minutes)
                    fb_verified, fb_report = self.verifier.verify(
                        fb_combined, context=context, forecasts=[fb]
                    )
                    warnings.extend(fb_report.warnings)
                    if fb_verified is not None:
                        verified = fb_verified
                        successful = [fb]
                        combined = fb_combined
                        fallback_used = True
                        fallback_reason = "combined forecast rejected by verifier; used safe baseline"
            if verified is None:
                return self._unavailable_result(
                    context, regime, excluded, combined.method, fallback_used, warnings,
                    results, trace, context_build_ms, t_total, fallback_reason, shadow,
                )
        verification_ms = (time.perf_counter() - t_ver) * 1000.0
        total_ms = (time.perf_counter() - t_total) * 1000.0 + context_build_ms

        diagnostics = self._diagnostics(
            context, trace, results, combined,
            context_build_ms=context_build_ms,
            combination_ms=combination_ms,
            verification_ms=verification_ms,
            total_ms=total_ms,
            fallback_reason=fallback_reason,
            route_source=trace.route_source,
            shadow=shadow,
        )
        topology = _trace_topology(results, verified.experts_used)
        telemetry = _trace_telemetry(results, trace, fallback_used)
        return ForecastResult(
            station_id=context.target_station_id,
            forecast_time_utc=str(context.forecast_time_utc),
            target_time_utc=str(context.target_time_utc),
            horizon_minutes=context.horizon_minutes,
            forecast_m=round(verified.forecast_m, 4),
            lower_m=round(verified.lower_m, 4),
            upper_m=round(verified.upper_m, 4),
            confidence=round(verified.confidence, 4),
            regime=regime,
            experts_used=verified.experts_used,
            experts_excluded=excluded,
            combination_method=verified.method,
            fallback_used=fallback_used,
            warnings=warnings,
            diagnostics=diagnostics,
            mode="mini",
            coordinator_policy_source="adaptive_cascade",
            coordinator_artifact_version="none",
            number_of_turns=telemetry["logical_actions"],
            number_of_unique_experts=telemetry["unique_experts"],
            role_sequence=topology["role_sequence"],
            executed_topology=topology["graph"],
            termination_reason="verified_combined_forecast",
            **telemetry,
        )

    # ------------------------------------------------------------------ #
    def _run_ultra(self, context) -> ForecastResult:
        conductor = UltraConductor(
            forecast_experts=self.experts,
            policy=self.ultra_policy,
            budget=self.ultra_budget,
        )
        run = conductor.run(context)
        state = run.state
        telemetry = state.telemetry().to_dict()
        transcript = [message.to_dict() for message in state.full_message_transcript]
        actions = [action.to_dict() for action in state.action_transcript]
        role_sequence = [message.role.value for message in state.full_message_transcript]
        excluded = {
            expert_id: "failed capability or safety mask at initialization"
            for expert_id, ok in state.capability_masks.items()
            if not ok
        }
        if run.candidate is None:
            warnings = [
                *run.warnings,
                "Ultra did not produce an independently accepted numerical forecast",
            ]
            return self._ultra_unavailable_result(
                context,
                excluded,
                warnings,
                state,
                telemetry,
                role_sequence,
                actions,
                transcript,
            )

        candidate = run.candidate
        return ForecastResult(
            station_id=context.target_station_id,
            forecast_time_utc=str(context.forecast_time_utc),
            target_time_utc=str(context.target_time_utc),
            horizon_minutes=context.horizon_minutes,
            forecast_m=round(candidate.forecast_m, 4),
            lower_m=round(candidate.lower_m, 4),
            upper_m=round(candidate.upper_m, 4),
            confidence=round(candidate.confidence, 4),
            regime=_ultra_regime(state),
            experts_used=list(candidate.experts_used),
            experts_excluded=excluded,
            combination_method=candidate.method,
            fallback_used="safe_fallback" in candidate.experts_used or telemetry.get("fallback_calls", 0) > 0,
            warnings=run.warnings,
            diagnostics={
                "context": context.diagnostics,
                "ultra": {
                    "actions": actions,
                    "transcript": transcript,
                    "workflow_graph": state.topology_dict(),
                    "verifier_findings": state.verifier_findings,
                    "state_features": state.encoded_state,
                    "candidate_diagnostics": candidate.diagnostics,
                    "telemetry": telemetry,
                },
            },
            status=run.status,
            mode="ultra",
            coordinator_policy_source=state.coordinator_policy_source,
            coordinator_artifact_version=state.coordinator_artifact_version,
            number_of_turns=telemetry["logical_actions"],
            number_of_unique_experts=telemetry["unique_experts"],
            role_sequence=role_sequence,
            executed_topology=state.topology_dict(),
            termination_reason=state.termination_reason or "completed",
            **telemetry,
        )

    def _ultra_unavailable_result(
        self,
        context,
        excluded,
        warnings,
        state,
        telemetry,
        role_sequence,
        actions,
        transcript,
    ) -> ForecastResult:
        return ForecastResult(
            station_id=context.target_station_id,
            forecast_time_utc=str(context.forecast_time_utc),
            target_time_utc=str(context.target_time_utc),
            horizon_minutes=context.horizon_minutes,
            forecast_m=None,
            lower_m=None,
            upper_m=None,
            confidence=0.0,
            regime="ultra_unavailable",
            experts_used=[],
            experts_excluded=excluded,
            combination_method="ultra_conductor",
            fallback_used=telemetry.get("fallback_calls", 0) > 0,
            warnings=warnings,
            diagnostics={
                "context": context.diagnostics,
                "ultra": {
                    "actions": actions,
                    "transcript": transcript,
                    "workflow_graph": state.topology_dict(),
                    "verifier_findings": state.verifier_findings,
                    "state_features": state.encoded_state,
                    "telemetry": telemetry,
                },
            },
            status="unavailable",
            mode="ultra",
            coordinator_policy_source=state.coordinator_policy_source,
            coordinator_artifact_version=state.coordinator_artifact_version,
            number_of_turns=telemetry["logical_actions"],
            number_of_unique_experts=telemetry["unique_experts"],
            role_sequence=role_sequence,
            executed_topology=state.topology_dict(),
            termination_reason=state.termination_reason or "unavailable",
            **telemetry,
        )

    # ------------------------------------------------------------------ #
    def _safe_fallback(self, context) -> ExpertForecast | None:
        expert = self.experts.get("safe_fallback")
        if expert is None:
            return None
        try:
            return expert.forecast(context)
        except Exception:  # noqa: BLE001
            return None

    def _skill_weights(self, context, forecasts, regime: str) -> dict[str, float]:
        return {
            f.model_name: self.skill_store.weight(
                expert=f.model_name,
                station=context.target_station_id,
                horizon_minutes=context.horizon_minutes,
                regime=regime,
            )
            for f in forecasts
        }

    def _skill_interval_floor(self, context, forecasts, regime: str) -> float | None:
        """Floor the combined interval at measured residual uncertainty.

        Only applies once enough validation samples exist; at cold start it
        returns ``None`` and the experts' own intervals are used unchanged.
        """

        best = None
        for f in forecasts:
            est = self.skill_store.estimate(
                expert=f.model_name,
                station=context.target_station_id,
                horizon_minutes=context.horizon_minutes,
                regime=regime,
            )
            if est.sample_count >= self.skill_store.min_samples:
                rmse_like = est.mae  # MAE is the available rolling metric
                cand = self.interval_inflation * rmse_like
                best = cand if best is None else max(best, cand)
        return best

    def _learned_shadow(self, context, trace) -> dict[str, Any] | None:
        if self.learned_router is None:
            return None
        try:
            shadow = self.learned_router.shadow_recommend(context)
            return {
                "would_select": shadow.recommended_expert,
                "rule_primary": trace.stage_1_expert,
                "agrees_with_rule": shadow.recommended_expert == trace.stage_1_expert,
                "source": shadow.source,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _diagnostics(self, context, trace, results, combined, *,
                     context_build_ms, combination_ms, verification_ms, total_ms,
                     fallback_reason, route_source, shadow) -> dict[str, Any]:
        a = trace.assessment
        return {
            "trace": {
                "context_build_ms": round(context_build_ms, 4),
                "capability_gate_ms": trace.capability_gate_ms,
                "routing_ms": trace.routing_ms,
                "per_expert_ms": trace.per_expert_ms,
                "combination_ms": round(combination_ms, 4),
                "verification_ms": round(verification_ms, 4),
                "total_ms": round(total_ms, 4),
                "expert_calls": trace.expert_calls,
                "cache_hits": trace.cache_hits,
                "stage_1_expert": trace.stage_1_expert,
                "escalated": trace.escalated,
                "escalation_reasons": trace.escalation_reasons,
                "early_stop_reason": trace.early_stop_reason,
                "timed_out_experts": trace.timed_out_experts,
                "route_source": route_source,
                "fallback_reason": fallback_reason,
                "execution_budget_used": trace.budget_used,
            },
            "assessment": asdict(a) if a is not None else None,
            "routing": {
                "regime": trace.regime,
                "primary": trace.plan.primary if trace.plan else None,
                "escalation_candidates": trace.plan.escalation_candidates if trace.plan else [],
                "ranked": [
                    {"name": r.name, "score": round(r.score, 4),
                     "predicted_error": round(r.predicted_error, 4),
                     "skill_samples": r.skill_samples, "skill_source": r.skill_source}
                    for r in (trace.plan.ranked if trace.plan else [])
                ],
            },
            "learned_router_shadow": shadow,
            "combiner": combined.diagnostics,
            "context": context.diagnostics,
            "experts": _forecast_diagnostics([r.forecast for r in results if r.forecast is not None]),
        }

    def _unavailable_result(self, context, regime, excluded, method, fallback_used,
                            warnings, results, trace, context_build_ms, t_total,
                            fallback_reason, shadow) -> ForecastResult:
        total_ms = (time.perf_counter() - t_total) * 1000.0 + context_build_ms
        diagnostics = {
            "trace": {
                "context_build_ms": round(context_build_ms, 4),
                "capability_gate_ms": trace.capability_gate_ms,
                "routing_ms": trace.routing_ms,
                "per_expert_ms": trace.per_expert_ms,
                "total_ms": round(total_ms, 4),
                "expert_calls": trace.expert_calls,
                "cache_hits": trace.cache_hits,
                "stage_1_expert": trace.stage_1_expert,
                "escalated": trace.escalated,
                "escalation_reasons": trace.escalation_reasons,
                "early_stop_reason": trace.early_stop_reason,
                "timed_out_experts": trace.timed_out_experts,
                "route_source": trace.route_source,
                "fallback_reason": fallback_reason,
                "execution_budget_used": trace.budget_used,
            },
            "learned_router_shadow": shadow,
            "context": context.diagnostics,
            "experts": _forecast_diagnostics([r.forecast for r in results if r.forecast is not None]),
        }
        return ForecastResult(
            station_id=context.target_station_id,
            forecast_time_utc=str(context.forecast_time_utc),
            target_time_utc=str(context.target_time_utc),
            horizon_minutes=context.horizon_minutes,
            forecast_m=None, lower_m=None, upper_m=None, confidence=0.0,
            regime=regime, experts_used=[], experts_excluded=excluded,
            combination_method=method, fallback_used=fallback_used,
            warnings=warnings, diagnostics=diagnostics, status="unavailable",
            mode="mini",
            coordinator_policy_source="adaptive_cascade",
            coordinator_artifact_version="none",
            number_of_turns=_trace_telemetry(results, trace, fallback_used)["logical_actions"],
            number_of_unique_experts=_trace_telemetry(results, trace, fallback_used)["unique_experts"],
            role_sequence=_trace_topology(results, [])["role_sequence"],
            executed_topology=_trace_topology(results, [])["graph"],
            termination_reason="unavailable",
            **_trace_telemetry(results, trace, fallback_used),
        )

    # ------------------------------------------------------------------ #
    # Legacy flat router (preserved for benchmarking and back-compat).
    # ------------------------------------------------------------------ #
    def _run_legacy(self, context, precomputed_forecasts) -> ForecastResult:
        decision = self.orchestrator.route(context)
        forecasts = self._run_selected(decision.selected_experts, context, precomputed_forecasts)
        valid, filter_report = self.verifier.filter_successful_experts(forecasts)

        fallback_used = decision.fallback_used
        if not valid and "safe_fallback" in self.experts and "safe_fallback" not in decision.selected_experts:
            fallback = self.experts["safe_fallback"].forecast(context)
            forecasts.append(fallback)
            fallback_used = True
            if fallback.ok:
                valid = [fallback]

        warnings = [*decision.warnings, *filter_report.warnings]
        excluded = dict(decision.excluded_experts)
        for removed in filter_report.removed_experts:
            excluded.setdefault(removed, "expert did not return a successful forecast")

        if not valid:
            warnings.append("No valid numerical forecast path is available")
            return self._legacy_unavailable(context, decision, excluded, fallback_used, warnings, forecasts)

        try:
            combined = self.combiner.combine(valid, method=decision.combination_method)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Combiner failed: {exc}")
            return self._legacy_unavailable(context, decision, excluded, fallback_used, warnings, forecasts)

        verified, verification = self.verifier.verify(combined, context=context, forecasts=valid)
        warnings.extend(verification.warnings)
        if verified is None:
            return self._legacy_unavailable(context, decision, excluded, fallback_used, warnings, forecasts)

        topology = _rule_topology(decision.selected_experts, verified.experts_used)
        telemetry = _rule_telemetry(decision.selected_experts, fallback_used)
        return ForecastResult(
            station_id=context.target_station_id,
            forecast_time_utc=str(context.forecast_time_utc),
            target_time_utc=str(context.target_time_utc),
            horizon_minutes=context.horizon_minutes,
            forecast_m=round(verified.forecast_m, 4),
            lower_m=round(verified.lower_m, 4),
            upper_m=round(verified.upper_m, 4),
            confidence=round(verified.confidence, 4),
            regime=decision.regime,
            experts_used=verified.experts_used,
            experts_excluded=excluded,
            combination_method=verified.method,
            fallback_used=fallback_used,
            warnings=warnings,
            diagnostics={
                "routing": {
                    "selected_experts": decision.selected_experts,
                    "confidence_adjustments": decision.confidence_adjustments,
                },
                "combiner": verified.diagnostics,
                "context": context.diagnostics,
                "experts": _forecast_diagnostics(forecasts),
            },
            mode="legacy",
            coordinator_policy_source="rule_based_router",
            coordinator_artifact_version="none",
            number_of_turns=telemetry["logical_actions"],
            number_of_unique_experts=telemetry["unique_experts"],
            role_sequence=topology["role_sequence"],
            executed_topology=topology["graph"],
            termination_reason="verified_combined_forecast",
            **telemetry,
        )

    def _run_selected(self, selected, context, precomputed_forecasts) -> list[ExpertForecast]:
        precomputed_forecasts = precomputed_forecasts or {}
        forecasts = []
        for name in selected:
            if name in precomputed_forecasts:
                forecasts.append(precomputed_forecasts[name])
                continue
            expert = self.experts.get(name)
            if expert is None:
                LOGGER.warning("Selected expert %s is not registered", name)
                continue
            try:
                forecasts.append(expert.forecast(context))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Expert %s failed", name)
                forecasts.append(expert.failed(context, str(exc)))
        return forecasts

    def _legacy_unavailable(self, context, decision, excluded, fallback_used, warnings, forecasts) -> ForecastResult:
        telemetry = _rule_telemetry([], fallback_used)
        return ForecastResult(
            station_id=context.target_station_id,
            forecast_time_utc=str(context.forecast_time_utc),
            target_time_utc=str(context.target_time_utc),
            horizon_minutes=context.horizon_minutes,
            forecast_m=None, lower_m=None, upper_m=None, confidence=0.0,
            regime=decision.regime, experts_used=[], experts_excluded=excluded,
            combination_method=decision.combination_method, fallback_used=fallback_used,
            warnings=warnings,
            diagnostics={
                "context": context.diagnostics,
                "experts": _forecast_diagnostics(forecasts),
            },
            status="unavailable",
            mode="legacy",
            coordinator_policy_source="rule_based_router",
            coordinator_artifact_version="none",
            number_of_turns=telemetry["logical_actions"],
            number_of_unique_experts=telemetry["unique_experts"],
            role_sequence=[],
            executed_topology={},
            termination_reason="unavailable",
            **telemetry,
        )


def _combination_method(successful: list[ExpertForecast]) -> str:
    n = len(successful)
    if n <= 1:
        return "single_expert"
    if n == 2:
        return "weighted_average"
    return "weighted_median"


def default_experts(*, include_placeholders: bool = False) -> dict[str, ForecastExpert]:
    experts: list[ForecastExpert] = [
        LocalPersistenceExpert(),
        LocalTideExpert(),
        NOAAResidualExpert(),
        RegionalToLocalResidualExpert(),
        HarmonicFallbackExpert(),
        SafeFallbackExpert(),
    ]
    if include_placeholders:
        experts.extend([
            WeatherAwareExpert(),
            SpatialNeighboringStationExpert(),
            LearnedLocalResidualExpert(),
        ])
    return {expert.model_name: expert for expert in experts}


def _forecast_diagnostics(forecasts: list[ExpertForecast]) -> dict[str, dict[str, Any]]:
    return {
        forecast.model_name: {
            "status": forecast.status,
            "message": forecast.message,
            "prediction_m": forecast.predicted_water_level_m,
            "confidence": forecast.confidence,
            "latency_ms": round(getattr(forecast, "latency_ms", 0.0), 4),
            "diagnostics": forecast.diagnostics,
        }
        for forecast in forecasts
    }


def _trace_telemetry(results, trace, fallback_used: bool) -> dict[str, int]:
    worker_calls = len(results)
    verifier_calls = 1 if results else 0
    return {
        "logical_actions": worker_calls + verifier_calls,
        "physical_expert_calls": int(getattr(trace, "expert_calls", worker_calls)),
        "reused_expert_outputs": int(getattr(trace, "cache_hits", 0)),
        "unique_experts": len({result.name for result in results}) + verifier_calls,
        "verifier_calls": verifier_calls,
        "thinker_calls": 0,
        "worker_calls": worker_calls,
        "fallback_calls": 1 if fallback_used else 0,
    }


def _trace_topology(results, accepted_experts: list[str]) -> dict[str, Any]:
    nodes = []
    access_edges = []
    dependency_edges = []
    for idx, result in enumerate(results):
        status = "success" if result.name in accepted_experts or result.ok else "failed"
        if getattr(result, "timed_out", False):
            status = "timeout"
        nodes.append({
            "turn_id": idx,
            "expert_id": result.name,
            "role": "WORKER",
            "subtask_kind": "MINI_ADAPTIVE_FORECAST",
            "status": status,
            "parallel_group": None,
            "latency_ms": float(getattr(result, "latency_ms", 0.0)),
        })
    verifier_turn = len(nodes)
    if nodes:
        nodes.append({
            "turn_id": verifier_turn,
            "expert_id": "forecast_verifier",
            "role": "VERIFIER",
            "subtask_kind": "MINI_VERIFY_COMBINED",
            "status": "accepted" if accepted_experts else "unavailable",
            "parallel_group": None,
            "latency_ms": 0.0,
        })
        for idx in range(verifier_turn):
            edge = {
                "source_turn_id": idx,
                "target_turn_id": verifier_turn,
                "edge_type": "mini_combiner_input",
            }
            access_edges.append(edge)
            dependency_edges.append(edge)
    return {
        "role_sequence": ["WORKER"] * len(results) + (["VERIFIER"] if nodes else []),
        "graph": {
            "nodes": nodes,
            "dependency_edges": dependency_edges,
            "access_edges": access_edges,
            "parallel_groups": {},
            "final_accepted_node": verifier_turn if accepted_experts else None,
        },
    }


def _rule_telemetry(selected_experts: list[str], fallback_used: bool) -> dict[str, int]:
    worker_calls = len(selected_experts)
    verifier_calls = 1 if selected_experts else 0
    return {
        "logical_actions": worker_calls + verifier_calls,
        "physical_expert_calls": worker_calls,
        "reused_expert_outputs": 0,
        "unique_experts": len(set(selected_experts)) + verifier_calls,
        "verifier_calls": verifier_calls,
        "thinker_calls": 0,
        "worker_calls": worker_calls,
        "fallback_calls": 1 if fallback_used else 0,
    }


def _rule_topology(selected_experts: list[str], accepted_experts: list[str]) -> dict[str, Any]:
    nodes = []
    access_edges = []
    dependency_edges = []
    for idx, expert_id in enumerate(selected_experts):
        nodes.append({
            "turn_id": idx,
            "expert_id": expert_id,
            "role": "WORKER",
            "subtask_kind": "LEGACY_RULE_FORECAST",
            "status": "success" if expert_id in accepted_experts else "filtered",
            "parallel_group": None,
            "latency_ms": 0.0,
        })
    verifier_turn = len(nodes)
    if selected_experts:
        nodes.append({
            "turn_id": verifier_turn,
            "expert_id": "forecast_verifier",
            "role": "VERIFIER",
            "subtask_kind": "LEGACY_VERIFY_COMBINED",
            "status": "accepted",
            "parallel_group": None,
            "latency_ms": 0.0,
        })
        for idx in range(len(selected_experts)):
            edge = {
                "source_turn_id": idx,
                "target_turn_id": verifier_turn,
                "edge_type": "legacy_combiner_input",
            }
            access_edges.append(edge)
            dependency_edges.append(edge)
    return {
        "role_sequence": ["WORKER"] * len(selected_experts) + (["VERIFIER"] if selected_experts else []),
        "graph": {
            "nodes": nodes,
            "dependency_edges": dependency_edges,
            "access_edges": access_edges,
            "parallel_groups": {},
            "final_accepted_node": verifier_turn if selected_experts else None,
        },
    }


def _ultra_regime(state) -> str:
    for message in reversed(state.full_message_transcript):
        if message.role.value == "THINKER":
            mechanism = message.structured_result.get("suspected_forcing_mechanism")
            if mechanism:
                return str(mechanism)
    if state.current_event_risk_estimate >= 0.35:
        return "event_risk"
    if state.current_difficulty_estimate >= 0.45:
        return "difficult_forecast"
    return "ultra_coordinated"
