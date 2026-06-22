"""Forecast verification and confidence adjustment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.experts.base import ExpertForecast
from src.orchestration.combiner import CombinedForecast
from src.orchestration.policies import VerificationPolicy


@dataclass
class VerificationReport:
    """Verifier decisions applied to a combined forecast."""

    status: str = "accepted"
    warnings: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    interval_multiplier: float = 1.0
    removed_experts: list[str] = field(default_factory=list)
    trigger_fallback: bool = False


class ForecastVerifier:
    """Check a combined forecast before it becomes user-facing output."""

    def __init__(self, policy: VerificationPolicy | None = None) -> None:
        self.policy = policy or VerificationPolicy()

    def filter_successful_experts(
        self,
        forecasts: list[ExpertForecast],
    ) -> tuple[list[ExpertForecast], VerificationReport]:
        report = VerificationReport()
        valid = []
        for forecast in forecasts:
            if forecast.ok:
                valid.append(forecast)
            else:
                report.removed_experts.append(forecast.model_name)
                if forecast.message:
                    report.warnings.append(f"{forecast.model_name}: {forecast.message}")
        if not valid:
            report.status = "unavailable"
            report.trigger_fallback = True
            report.warnings.append("No successful expert forecast remains after filtering")
        return valid, report

    def verify(
        self,
        combined: CombinedForecast,
        *,
        context,
        forecasts: list[ExpertForecast],
    ) -> tuple[CombinedForecast | None, VerificationReport]:
        report = VerificationReport()

        if combined.lower_m > combined.upper_m:
            report.status = "unavailable"
            report.warnings.append("Invalid uncertainty interval: lower bound exceeds upper bound")
            return None, report

        if not (combined.lower_m <= combined.forecast_m <= combined.upper_m):
            report.warnings.append("Forecast was outside interval; interval widened to include point forecast")
            combined.lower_m = min(combined.lower_m, combined.forecast_m)
            combined.upper_m = max(combined.upper_m, combined.forecast_m)
            report.confidence_delta -= 0.05

        if not (self.policy.plausible_min_m <= combined.forecast_m <= self.policy.plausible_max_m):
            report.status = "unavailable"
            report.warnings.append("Forecast is outside plausible configured water-level range")
            return None, report

        for source, freshness in context.observation_freshness_seconds.items():
            if freshness > self.policy.max_input_staleness_seconds:
                report.warnings.append(f"{source} input is stale ({freshness:.0f}s old)")
                report.confidence_delta -= 0.1

        if not context.hohonu_qc_ok:
            report.warnings.append("Hohonu QC status is not good")
            report.confidence_delta -= 0.1
        if not context.noaa_qc_ok and any(f.model_name in {"noaa_residual", "regional_to_local_residual"} for f in forecasts):
            report.warnings.append("NOAA QC status is not good")
            report.confidence_delta -= 0.1

        latest = context.latest_hohonu_observation
        if latest is not None:
            jump = abs(combined.forecast_m - float(latest["water_level_m"]))
            if jump > self.policy.suspicious_jump_m:
                report.warnings.append(f"Physically suspicious jump from latest local observation ({jump:.2f} m)")
                report.confidence_delta -= 0.2
                report.interval_multiplier = max(report.interval_multiplier, 1.5)

        successful_values = [
            float(f.predicted_water_level_m) for f in forecasts if f.ok
        ]
        if len(successful_values) >= 2:
            disagreement = float(np.max(successful_values) - np.min(successful_values))
            if disagreement > self.policy.high_disagreement_m:
                report.warnings.append(f"High model disagreement ({disagreement:.2f} m)")
                report.confidence_delta -= 0.15
                report.interval_multiplier = max(
                    report.interval_multiplier,
                    self.policy.disagreement_interval_multiplier,
                )

        if context.noaa_tide_prediction is None and not any(f.model_name == "safe_fallback" and f.ok for f in forecasts):
            report.warnings.append("No safe tide baseline is available")
            report.confidence_delta -= 0.15

        if report.interval_multiplier > 1.0:
            half = max(
                combined.forecast_m - combined.lower_m,
                combined.upper_m - combined.forecast_m,
            ) * report.interval_multiplier
            combined.lower_m = float(combined.forecast_m - half)
            combined.upper_m = float(combined.forecast_m + half)

        combined.confidence = max(0.0, min(1.0, combined.confidence + report.confidence_delta))
        if combined.confidence <= 0.05:
            report.status = "unavailable"
            report.warnings.append("Confidence fell below the availability threshold")
            return None, report

        return combined, report
