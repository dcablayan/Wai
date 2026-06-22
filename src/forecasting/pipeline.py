"""End-to-end orchestrated forecasting pipeline."""

from __future__ import annotations

import logging
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
from src.orchestration.combiner import ForecastCombiner
from src.orchestration.router import RuleBasedOrchestrator
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
    ) -> None:
        self.experts = experts or default_experts(include_placeholders=False)
        self.orchestrator = orchestrator or RuleBasedOrchestrator()
        self.combiner = combiner or ForecastCombiner()
        self.verifier = verifier or ForecastVerifier()

    def run(self, context) -> ForecastResult:
        decision = self.orchestrator.route(context)
        LOGGER.info(
            "Routing station %s horizon %smin to %s",
            context.target_station_id,
            context.horizon_minutes,
            decision.selected_experts,
        )

        forecasts = self._run_selected(decision.selected_experts, context)
        valid, filter_report = self.verifier.filter_successful_experts(forecasts)

        fallback_used = decision.fallback_used
        if not valid and "safe_fallback" in self.experts and "safe_fallback" not in decision.selected_experts:
            fallback = self.experts["safe_fallback"].forecast(context)
            forecasts.append(fallback)
            fallback_used = True
            if fallback.ok:
                valid = [fallback]

        warnings = [
            *decision.warnings,
            *filter_report.warnings,
        ]
        excluded = dict(decision.excluded_experts)
        for removed in filter_report.removed_experts:
            excluded.setdefault(removed, "expert did not return a successful forecast")

        if not valid:
            warnings.append("No valid numerical forecast path is available")
            return self._unavailable_result(
                context,
                decision.regime,
                excluded,
                decision.combination_method,
                fallback_used,
                warnings,
                forecasts,
            )

        try:
            combined = self.combiner.combine(valid, method=decision.combination_method)
        except Exception as exc:
            warnings.append(f"Combiner failed: {exc}")
            return self._unavailable_result(
                context,
                decision.regime,
                excluded,
                decision.combination_method,
                fallback_used,
                warnings,
                forecasts,
            )

        verified, verification = self.verifier.verify(combined, context=context, forecasts=valid)
        warnings.extend(verification.warnings)
        if verified is None:
            return self._unavailable_result(
                context,
                decision.regime,
                excluded,
                decision.combination_method,
                fallback_used,
                warnings,
                forecasts,
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

    def _run_selected(self, selected: list[str], context) -> list[ExpertForecast]:
        forecasts = []
        for name in selected:
            expert = self.experts.get(name)
            if expert is None:
                LOGGER.warning("Selected expert %s is not registered", name)
                continue
            try:
                forecasts.append(expert.forecast(context))
            except Exception as exc:
                LOGGER.exception("Expert %s failed", name)
                forecasts.append(expert.failed(context, str(exc)))
        return forecasts

    def _unavailable_result(
        self,
        context,
        regime: str,
        excluded: dict[str, str],
        combination_method: str,
        fallback_used: bool,
        warnings: list[str],
        forecasts: list[ExpertForecast],
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
            regime=regime,
            experts_used=[],
            experts_excluded=excluded,
            combination_method=combination_method,
            fallback_used=fallback_used,
            warnings=warnings,
            diagnostics={
                "context": context.diagnostics,
                "experts": _forecast_diagnostics(forecasts),
            },
            status="unavailable",
        )


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
            "diagnostics": forecast.diagnostics,
        }
        for forecast in forecasts
    }
