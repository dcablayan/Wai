"""Rule-based expert router for the first Wai orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.orchestration.policies import RoutingPolicy


@dataclass
class RoutingDecision:
    """Which experts to run and why."""

    regime: str
    selected_experts: list[str]
    excluded_experts: dict[str, str] = field(default_factory=dict)
    combination_method: str = "weighted_median"
    fallback_used: bool = False
    confidence_adjustments: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RuleBasedOrchestrator:
    """Select one to three numerical experts from current conditions."""

    def __init__(self, policy: RoutingPolicy | None = None) -> None:
        self.policy = policy or RoutingPolicy()

    def route(self, context) -> RoutingDecision:
        excluded: dict[str, str] = {}
        warnings: list[str] = []
        selected: list[str] = []
        regime = "normal_tide_residual"

        hohonu_fresh = (
            context.observation_freshness_seconds.get("hohonu", float("inf"))
            <= self.policy.fresh_hohonu_seconds
        )
        noaa_fresh = (
            context.observation_freshness_seconds.get("noaa", float("inf"))
            <= self.policy.fresh_noaa_seconds
        )
        noaa_residual = abs(context.recent_noaa_residual_m or 0.0)

        if not context.hohonu_qc_ok:
            excluded["local_persistence"] = "latest Hohonu observation failed QC"
        elif not hohonu_fresh:
            excluded["local_persistence"] = "latest Hohonu observation is stale"

        if not context.noaa_qc_ok:
            excluded["noaa_residual"] = "latest NOAA observation failed QC"
            excluded["regional_to_local_residual"] = "latest NOAA observation failed QC"
        elif not noaa_fresh:
            excluded["noaa_residual"] = "latest NOAA observation is stale"
            excluded["regional_to_local_residual"] = "latest NOAA observation is stale"

        if context.noaa_tide_prediction is None:
            excluded["local_tide"] = "tide prediction is missing"
            excluded["safe_fallback"] = "safe tide baseline is missing"

        if context.horizon_minutes <= self.policy.short_horizon_minutes and "local_persistence" not in excluded:
            regime = "fresh_local_short_horizon"
            selected.append("local_persistence")

        if noaa_residual >= self.policy.large_noaa_residual_m and "noaa_residual" not in excluded:
            regime = "regional_non_tidal_event"
            selected.extend(["noaa_residual", "regional_to_local_residual"])

        if not selected:
            if "local_tide" not in excluded:
                selected.append("local_tide")
            if context.horizon_minutes <= self.policy.normal_horizon_minutes and "noaa_residual" not in excluded:
                selected.append("noaa_residual")

        if not noaa_fresh and "local_persistence" not in excluded:
            regime = "stale_noaa_local_tide"
            _append_unique(selected, "local_persistence")
            warnings.append("NOAA observations are stale; local observations are preferred")

        if not context.hohonu_qc_ok and "safe_fallback" not in excluded:
            regime = "failed_local_qc_safe_fallback"
            _append_unique(selected, "safe_fallback")
            warnings.append("Hohonu QC failed; local persistence excluded")

        if (
            context.model_disagreement_m is not None
            and context.model_disagreement_m >= self.policy.strong_disagreement_m
        ):
            _append_unique(selected, "regional_to_local_residual")
            warnings.append("Strong model disagreement detected; additional expert requested")

        if not selected and "safe_fallback" not in excluded:
            selected.append("safe_fallback")

        selected = [
            name for name in selected
            if name not in excluded
        ][: self.policy.max_selected_experts]

        fallback_used = selected == ["safe_fallback"] or "safe_fallback" in selected
        if not selected:
            warnings.append("No routeable experts passed data-availability checks")

        return RoutingDecision(
            regime=regime,
            selected_experts=selected,
            excluded_experts=excluded,
            combination_method=self.policy.default_combination_method,
            fallback_used=fallback_used,
            warnings=warnings,
        )


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)
