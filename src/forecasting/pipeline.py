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
from src.orchestration.router import RuleBasedOrchestrator
from src.orchestration.skill_store import SkillStore
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ForecastPipeline:
    """Rule-based expert orchestration, combination, and verification."""

    def __init__(
        self,
        experts: dict[str, ForecastExpert] | None = None,
        orchestrator: RuleBasedOrchestrator | None = None,
        combiner: ForecastCombiner | None = None,
        verifier: ForecastVerifier | None = None,
        *,
        adaptive: bool = True,
        cascade: AdaptiveCascade | None = None,
        skill_store: SkillStore | None = None,
        budget: ExecutionBudget | None = None,
        learned_router: Any | None = None,
        interval_inflation: float = 1.8,
    ) -> None:
        self.experts = experts or default_experts(include_placeholders=False)
        self.orchestrator = orchestrator or RuleBasedOrchestrator()
        self.combiner = combiner or ForecastCombiner()
        self.verifier = verifier or ForecastVerifier()
        self.adaptive = adaptive
        self.skill_store = skill_store or SkillStore()
        self.cascade = cascade or AdaptiveCascade(skill_store=self.skill_store)
        self.budget = budget or ExecutionBudget()
        self.learned_router = learned_router
        self.interval_inflation = interval_inflation

    # ------------------------------------------------------------------ #
    def run(
        self,
        context,
        *,
        precomputed_forecasts: dict[str, ExpertForecast] | None = None,
        budget: ExecutionBudget | None = None,
        context_build_ms: float = 0.0,
    ) -> ForecastResult:
        if not self.adaptive:
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
