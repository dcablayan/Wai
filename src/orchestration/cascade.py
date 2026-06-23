"""Adaptive forecast cascade.

Replaces the flat, single-shot ``selected_experts`` list with a staged plan:

    capability gate -> primary ranking -> Stage 1 cheap forecast
    -> post-forecast assessment -> early stop OR conditional escalation
    -> optional Stage 2 experts

Key correctness properties versus the first router:

- Impossible experts are excluded *before* execution using declarative
  :class:`ExpertSpec` metadata, not by running them and waiting to fail.
- Model disagreement is computed from *actual* completed forecasts, never from a
  context field populated before any expert has run.
- Recent measured skill (from a :class:`SkillStore`) feeds an interpretable,
  cost-aware route score; hard data/QC/datum gates remain authoritative.
- A near-zero-cost tide baseline is read from the context, not produced by
  running a full expert, and the safe-fallback call can be reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.experts.base import ExpertForecast, ForecastExpert
from src.orchestration.executor import ExpertRun, run_experts
from src.orchestration.skill_store import SkillStore


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutionBudget:
    """Bounds on how much compute one forecast may spend."""

    deadline_ms: float | None = None
    max_expert_calls: int = 3
    max_parallelism: int = 2
    per_expert_timeout_ms: float | None = None
    reserve_fallback_call: bool = True
    allow_parallel_escalation: bool = False


@dataclass(frozen=True)
class CascadePolicy:
    """Thresholds governing early-stop and escalation decisions."""

    latency_weight: float = 0.01
    failure_weight: float = 0.5
    safety_penalty: float = 0.25
    min_confidence_early_stop: float = 0.6
    max_interval_width_early_stop: float = 0.6
    max_baseline_diff_early_stop: float = 0.30
    strong_baseline_disagreement_m: float = 0.30
    wide_interval_m: float = 0.7
    low_confidence: float = 0.5
    event_risk_residual_m: float = 0.25
    ood_residual_m: float = 0.6
    suspicious_jump_m: float = 1.5
    max_escalation_experts: int = 2
    degraded_failure_rate: float = 0.4


# --------------------------------------------------------------------------- #
# Capability gate
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityReport:
    eligible: list[str]
    excluded: dict[str, str]


class CapabilityGate:
    """Exclude experts that cannot possibly run for this context."""

    def __init__(self, *, fresh_local_seconds: float = 3 * 60 * 60, fresh_noaa_seconds: float = 3 * 60 * 60) -> None:
        self.fresh_local_seconds = fresh_local_seconds
        self.fresh_noaa_seconds = fresh_noaa_seconds

    def evaluate(self, context, experts: dict[str, ForecastExpert]) -> CapabilityReport:
        eligible: list[str] = []
        excluded: dict[str, str] = {}
        tide_available = (
            context.noaa_tide_prediction is not None or context.local_tide_prediction is not None
        )
        local_fresh = (
            context.observation_freshness_seconds.get("hohonu", float("inf"))
            <= self.fresh_local_seconds
        )
        noaa_fresh = (
            context.observation_freshness_seconds.get("noaa", float("inf"))
            <= self.fresh_noaa_seconds
        )
        for name, expert in experts.items():
            spec = getattr(expert, "spec", None)
            reason = self._exclusion_reason(
                spec, context, tide_available, local_fresh, noaa_fresh
            )
            if reason:
                excluded[name] = reason
            else:
                eligible.append(name)
        return CapabilityReport(eligible=eligible, excluded=excluded)

    def _exclusion_reason(self, spec, context, tide_available, local_fresh, noaa_fresh) -> str | None:
        if spec is None:
            return None
        if not spec.supports_horizon(context.horizon_minutes):
            return (
                f"horizon {context.horizon_minutes}min outside supported range "
                f"[{spec.min_horizon_minutes}, {spec.max_horizon_minutes}]"
            )
        if spec.requires_tide and not tide_available:
            return "tide prediction is missing"
        if spec.requires_local_obs:
            if context.latest_hohonu_observation is None:
                return "latest Hohonu observation is missing"
            if not context.hohonu_qc_ok:
                return "latest Hohonu observation failed QC"
            if not local_fresh:
                return "latest Hohonu observation is stale"
        if spec.requires_noaa_obs:
            if context.latest_noaa_observation is None:
                return "latest NOAA observation is missing"
            if context.recent_noaa_residual_m is None:
                return "recent NOAA residual is missing"
            if not context.noaa_qc_ok:
                return "latest NOAA observation failed QC"
            if not noaa_fresh:
                return "latest NOAA observation is stale"
        if "weather_observation" in spec.required_sources and context.wind_speed_mps is None and context.pressure_trend is None:
            return "weather observations are unavailable"
        if "neighboring_station" in spec.required_sources and not context.neighboring_station_signals:
            return "neighboring-station signals are unavailable"
        return None


# --------------------------------------------------------------------------- #
# Plan / results / assessment / trace
# --------------------------------------------------------------------------- #
@dataclass
class RankedExpert:
    name: str
    score: float
    predicted_error: float
    expected_latency: float
    failure_rate: float
    skill_samples: int
    skill_source: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class ForecastPlan:
    regime: str
    primary: str | None
    escalation_candidates: list[str]
    baseline_expert: str | None
    ranked: list[RankedExpert]
    excluded: dict[str, str]


@dataclass
class ExpertExecutionResult:
    name: str
    stage: str
    forecast: ExpertForecast | None
    latency_ms: float
    reused: bool = False
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.forecast is not None and self.forecast.ok


@dataclass
class PostForecastAssessment:
    success: bool
    confidence: float
    interval_width_m: float | None
    baseline_value_m: float | None
    baseline_diff_m: float | None
    disagreement_m: float | None
    input_quality: str
    recent_skill_mae: float | None
    recent_skill_samples: int
    out_of_distribution: bool
    suspicious_jump: bool
    elevated_event_risk: bool
    remaining_calls: int


@dataclass
class ExecutionTrace:
    regime: str
    stage_1_expert: str | None = None
    escalated: bool = False
    escalation_reasons: list[str] = field(default_factory=list)
    early_stop_reason: str | None = None
    route_source: str = "rule_cascade"
    expert_calls: int = 0
    cache_hits: int = 0
    timed_out_experts: list[str] = field(default_factory=list)
    per_expert_ms: dict[str, float] = field(default_factory=dict)
    capability_gate_ms: float = 0.0
    routing_ms: float = 0.0
    execution_results: list[ExpertExecutionResult] = field(default_factory=list)
    plan: ForecastPlan | None = None
    assessment: PostForecastAssessment | None = None
    budget_used: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Cascade
# --------------------------------------------------------------------------- #
class AdaptiveCascade:
    """Plan and execute experts with early stop and conditional escalation."""

    def __init__(
        self,
        *,
        gate: CapabilityGate | None = None,
        policy: CascadePolicy | None = None,
        skill_store: SkillStore | None = None,
    ) -> None:
        self.gate = gate or CapabilityGate()
        self.policy = policy or CascadePolicy()
        self.skill_store = skill_store or SkillStore()

    # -- planning -----------------------------------------------------------
    def plan(self, context, experts: dict[str, ForecastExpert], capability: CapabilityReport) -> ForecastPlan:
        regime = _detect_regime(context, self.policy)
        ranked: list[RankedExpert] = []
        baseline_expert = None
        for name in capability.eligible:
            spec = getattr(experts[name], "spec", None)
            est = self.skill_store.estimate(
                expert=name,
                station=context.target_station_id,
                horizon_minutes=context.horizon_minutes,
                regime=regime,
            )
            expected_latency = spec.expected_latency_units if spec else 5.0
            is_safe = bool(spec and spec.is_safe_baseline)
            if is_safe:
                baseline_expert = name
            adj, reasons = _regime_adjustment(name, context, regime, self.policy)
            score = (
                est.mae
                + self.policy.latency_weight * expected_latency
                + self.policy.failure_weight * est.failure_rate
                + (self.policy.safety_penalty if is_safe else 0.0)
                + adj
            )
            ranked.append(
                RankedExpert(
                    name=name,
                    score=score,
                    predicted_error=est.mae,
                    expected_latency=expected_latency,
                    failure_rate=est.failure_rate,
                    skill_samples=est.sample_count,
                    skill_source=est.source_level,
                    reasons=reasons,
                )
            )
        ranked.sort(key=lambda r: r.score)
        non_baseline = [
            r for r in ranked
            if not getattr(experts[r.name], "spec", None) or not experts[r.name].spec.is_safe_baseline
        ]
        primary = non_baseline[0].name if non_baseline else (baseline_expert or (ranked[0].name if ranked else None))
        escalation_candidates = [r.name for r in non_baseline[1:]]
        return ForecastPlan(
            regime=regime,
            primary=primary,
            escalation_candidates=escalation_candidates,
            baseline_expert=baseline_expert,
            ranked=ranked,
            excluded=dict(capability.excluded),
        )

    # -- assessment ---------------------------------------------------------
    def assess(
        self,
        context,
        plan: ForecastPlan,
        completed: list[ExpertExecutionResult],
        remaining_calls: int,
    ) -> PostForecastAssessment:
        ok_forecasts = [r.forecast for r in completed if r.ok]
        primary_fc = ok_forecasts[0] if ok_forecasts else None
        baseline_value = _baseline_value(context)

        confidence = float(primary_fc.confidence) if primary_fc else 0.0
        interval_width = (
            float(primary_fc.upper_m - primary_fc.lower_m) if primary_fc else None
        )
        # Disagreement from the tide baseline is only a meaningful signal when the
        # primary forecast is built on (or directly comparable to) the tide.  A
        # pure-local expert such as persistence legitimately sits at a local datum
        # offset from a *regional* tide; its sanity is guarded by suspicious_jump,
        # not by the regional baseline.  Skip the comparison when it is confounded.
        baseline_comparable = primary_fc is not None and (
            _is_tide_referenced(primary_fc.model_name)
            or context.local_tide_prediction is not None
        )
        baseline_diff = (
            abs(float(primary_fc.predicted_water_level_m) - baseline_value)
            if baseline_comparable and baseline_value is not None
            else None
        )
        # Real disagreement across all completed successful forecasts.
        values = [float(f.predicted_water_level_m) for f in ok_forecasts]
        disagreement = (max(values) - min(values)) if len(values) >= 2 else None

        residual = abs(context.recent_noaa_residual_m or 0.0)
        residual_trend = abs(context.noaa_residual_trend_m_per_hour or 0.0)
        elevated_event_risk = (
            residual >= self.policy.event_risk_residual_m
            or residual_trend >= 0.15
        )
        ood = residual > self.policy.ood_residual_m
        suspicious_jump = False
        if primary_fc is not None and context.latest_hohonu_observation is not None:
            jump = abs(
                float(primary_fc.predicted_water_level_m)
                - float(context.latest_hohonu_observation["water_level_m"])
            )
            suspicious_jump = jump > self.policy.suspicious_jump_m

        input_quality = _input_quality(context)
        skill_mae = None
        skill_samples = 0
        if primary_fc is not None:
            est = self.skill_store.estimate(
                expert=primary_fc.model_name,
                station=context.target_station_id,
                horizon_minutes=context.horizon_minutes,
                regime=plan.regime,
            )
            skill_mae = est.mae
            skill_samples = est.sample_count

        return PostForecastAssessment(
            success=primary_fc is not None,
            confidence=confidence,
            interval_width_m=interval_width,
            baseline_value_m=baseline_value,
            baseline_diff_m=baseline_diff,
            disagreement_m=disagreement,
            input_quality=input_quality,
            recent_skill_mae=skill_mae,
            recent_skill_samples=skill_samples,
            out_of_distribution=ood,
            suspicious_jump=suspicious_jump,
            elevated_event_risk=elevated_event_risk,
            remaining_calls=remaining_calls,
        )

    def should_escalate(self, assessment: PostForecastAssessment, plan: ForecastPlan) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not assessment.success:
            reasons.append("primary expert failed")
            return True, reasons
        if assessment.confidence < self.policy.low_confidence:
            reasons.append(f"low confidence ({assessment.confidence:.2f})")
        if assessment.interval_width_m is not None and assessment.interval_width_m > self.policy.wide_interval_m:
            reasons.append(f"wide interval ({assessment.interval_width_m:.2f} m)")
        if (
            assessment.baseline_diff_m is not None
            and assessment.baseline_diff_m > self.policy.strong_baseline_disagreement_m
        ):
            reasons.append(f"strong disagreement with tide baseline ({assessment.baseline_diff_m:.2f} m)")
        if assessment.elevated_event_risk:
            reasons.append("elevated event risk (abnormal NOAA residual/trend)")
        if assessment.out_of_distribution:
            reasons.append("out-of-distribution residual magnitude")
        if assessment.suspicious_jump:
            reasons.append("suspicious jump from latest local observation")
        if assessment.recent_skill_samples >= 5 and assessment.recent_skill_mae is not None:
            # Degraded recent primary performance justifies a second opinion.
            est_fail = None  # captured via plan ranked
            for r in plan.ranked:
                if r.name == plan.primary:
                    est_fail = r.failure_rate
            if est_fail is not None and est_fail >= self.policy.degraded_failure_rate:
                reasons.append("recent primary-expert performance degraded")
        escalate = bool(reasons) and assessment.remaining_calls > 0
        return escalate, reasons

    def early_stop_reason(self, assessment: PostForecastAssessment) -> str | None:
        if not assessment.success:
            return None
        if assessment.confidence < self.policy.min_confidence_early_stop:
            return None
        if assessment.interval_width_m is not None and assessment.interval_width_m > self.policy.max_interval_width_early_stop:
            return None
        if assessment.baseline_diff_m is not None and assessment.baseline_diff_m > self.policy.max_baseline_diff_early_stop:
            return None
        if assessment.elevated_event_risk or assessment.out_of_distribution or assessment.suspicious_jump:
            return None
        return (
            "primary forecast confident, well-bounded, agrees with tide baseline, "
            "and operating in a well-supported regime"
        )

    # -- execution ----------------------------------------------------------
    def execute(
        self,
        context,
        experts: dict[str, ForecastExpert],
        *,
        budget: ExecutionBudget,
        precomputed: dict[str, ExpertForecast] | None = None,
    ) -> tuple[list[ExpertExecutionResult], ExecutionTrace]:
        import time

        precomputed = precomputed or {}
        t_gate = time.perf_counter()
        capability = self.gate.evaluate(context, experts)
        gate_ms = (time.perf_counter() - t_gate) * 1000.0

        t_route = time.perf_counter()
        plan = self.plan(context, experts, capability)
        routing_ms = (time.perf_counter() - t_route) * 1000.0

        trace = ExecutionTrace(
            regime=plan.regime,
            stage_1_expert=plan.primary,
            plan=plan,
            capability_gate_ms=round(gate_ms, 4),
            routing_ms=round(routing_ms, 4),
        )

        reserved = 1 if budget.reserve_fallback_call else 0
        # Budget for optional (non-fallback) numerical expert calls.
        call_budget = max(0, budget.max_expert_calls - reserved if plan.baseline_expert else budget.max_expert_calls)
        # Always allow at least the primary call.
        call_budget = max(call_budget, 1)
        results: list[ExpertExecutionResult] = []
        calls_used = 0

        if plan.primary is None:
            trace.assessment = self.assess(context, plan, results, 0)
            trace.budget_used = _budget_used(budget, calls_used, reserved)
            return results, trace

        # Stage 1: primary
        primary_result = self._run_one(plan.primary, experts, context, "stage_1_primary", precomputed, budget)
        results.append(primary_result)
        if not primary_result.reused:
            calls_used += 1
        trace.expert_calls = calls_used
        trace.cache_hits += int(primary_result.reused)
        trace.per_expert_ms[plan.primary] = round(primary_result.latency_ms, 4)
        if primary_result.timed_out:
            trace.timed_out_experts.append(plan.primary)

        remaining = call_budget - calls_used
        assessment = self.assess(context, plan, results, remaining)
        trace.assessment = assessment

        escalate, reasons = self.should_escalate(assessment, plan)
        if not escalate:
            trace.early_stop_reason = self.early_stop_reason(assessment) or "no escalation trigger met"
            trace.budget_used = _budget_used(budget, calls_used, reserved)
            trace.execution_results = results
            return results, trace

        # Stage 2: conditional escalation
        trace.escalated = True
        trace.escalation_reasons = reasons
        n_extra = min(
            self.policy.max_escalation_experts,
            remaining,
            len(plan.escalation_candidates),
        )
        to_run = [
            name for name in plan.escalation_candidates
            if name not in {r.name for r in results}
        ][:n_extra]

        if to_run:
            stage2_results = self._run_many(
                to_run, experts, context, "stage_2_escalation", precomputed, budget
            )
            for r in stage2_results:
                results.append(r)
                if not r.reused:
                    calls_used += 1
                trace.cache_hits += int(r.reused)
                trace.per_expert_ms[r.name] = round(r.latency_ms, 4)
                if r.timed_out:
                    trace.timed_out_experts.append(r.name)

        trace.expert_calls = calls_used
        # Re-assess with all completed forecasts (real disagreement now available).
        trace.assessment = self.assess(context, plan, results, max(0, call_budget - calls_used))
        trace.budget_used = _budget_used(budget, calls_used, reserved)
        trace.execution_results = results
        return results, trace

    def _run_one(self, name, experts, context, stage, precomputed, budget) -> ExpertExecutionResult:
        if name in precomputed:
            fc = precomputed[name]
            return ExpertExecutionResult(
                name=name, stage=stage, forecast=fc,
                latency_ms=getattr(fc, "latency_ms", 0.0), reused=True,
            )
        run = run_experts(
            [experts[name]], context,
            parallel=False,
            per_expert_timeout_ms=budget.per_expert_timeout_ms,
        )[0]
        return _to_exec_result(run, stage)

    def _run_many(self, names, experts, context, stage, precomputed, budget) -> list[ExpertExecutionResult]:
        out: list[ExpertExecutionResult] = []
        to_execute = []
        for name in names:
            if name in precomputed:
                fc = precomputed[name]
                out.append(ExpertExecutionResult(
                    name=name, stage=stage, forecast=fc,
                    latency_ms=getattr(fc, "latency_ms", 0.0), reused=True,
                ))
            else:
                to_execute.append(name)
        if to_execute:
            parallel = budget.allow_parallel_escalation and len(to_execute) > 1
            # Only parallelize thread-safe experts.
            safe = all(getattr(experts[n], "spec", None) and experts[n].spec.thread_safe for n in to_execute)
            runs = run_experts(
                [experts[n] for n in to_execute], context,
                parallel=parallel and safe,
                max_parallelism=budget.max_parallelism,
                per_expert_timeout_ms=budget.per_expert_timeout_ms,
            )
            out.extend(_to_exec_result(run, stage) for run in runs)
        # Preserve requested order.
        order = {name: i for i, name in enumerate(names)}
        out.sort(key=lambda r: order.get(r.name, 0))
        return out


def _to_exec_result(run: ExpertRun, stage: str) -> ExpertExecutionResult:
    return ExpertExecutionResult(
        name=run.name,
        stage=stage,
        forecast=run.forecast,
        latency_ms=run.latency_ms,
        timed_out=run.timed_out,
        error=run.error,
    )


def _budget_used(budget: ExecutionBudget, calls_used: int, reserved: int) -> dict[str, Any]:
    return {
        "max_expert_calls": budget.max_expert_calls,
        "expert_calls_used": calls_used,
        "fallback_call_reserved": bool(reserved),
        "deadline_ms": budget.deadline_ms,
        "parallel_escalation": budget.allow_parallel_escalation,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_TIDE_REFERENCED_EXPERTS = {
    "local_tide",
    "noaa_residual",
    "regional_to_local_residual",
    "safe_fallback",
}


def _is_tide_referenced(model_name: str) -> bool:
    return model_name in _TIDE_REFERENCED_EXPERTS


def _baseline_value(context) -> float | None:
    tide = context.local_tide_prediction or context.noaa_tide_prediction
    if not tide:
        return None
    return float(tide["water_level_m"])


def _detect_regime(context, policy: CascadePolicy) -> str:
    residual = abs(context.recent_noaa_residual_m or 0.0)
    if not context.hohonu_qc_ok:
        return "failed_local_qc_safe_fallback"
    if residual >= policy.event_risk_residual_m:
        return "regional_non_tidal_event"
    if context.horizon_minutes <= 90 and context.hohonu_is_fresh:
        return "fresh_local_short_horizon"
    if not context.noaa_is_fresh:
        return "stale_noaa_local_tide"
    return "normal_tide_residual"


def _regime_adjustment(name: str, context, regime: str, policy: CascadePolicy) -> tuple[float, list[str]]:
    """Interpretable, horizon/regime-aware score nudges (lower is better)."""

    reasons: list[str] = []
    adj = 0.0
    horizon_hours = context.horizon_minutes / 60.0
    if name == "local_persistence":
        # Persistence degrades past ~90 minutes.
        penalty = 0.03 * max(0.0, horizon_hours - 1.5)
        if penalty:
            adj += penalty
            reasons.append("persistence penalized beyond short horizon")
        if regime == "fresh_local_short_horizon":
            adj -= 0.05
            reasons.append("fresh local data favors persistence at short horizon")
    if name in {"noaa_residual", "regional_to_local_residual"} and regime == "regional_non_tidal_event":
        adj -= 0.05
        reasons.append("large NOAA residual favors residual experts")
    if name == "local_tide" and regime == "stale_noaa_local_tide":
        adj -= 0.03
        reasons.append("stale NOAA favors tide path")
    return adj, reasons


def _input_quality(context) -> str:
    if not context.hohonu_qc_ok or not context.noaa_qc_ok:
        return "degraded"
    if not context.hohonu_is_fresh and not context.noaa_is_fresh:
        return "stale"
    return "good"
