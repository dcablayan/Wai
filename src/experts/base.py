"""Common interface for numerical forecasting experts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from src.experts.capabilities import ExpertSpec


SUCCESS = "success"
FAILED = "failed"
UNAVAILABLE = "unavailable"


@dataclass
class ExpertForecast:
    """Forecast emitted by one numerical expert."""

    model_name: str
    forecast_time_utc: pd.Timestamp
    target_time_utc: pd.Timestamp
    horizon_minutes: int
    predicted_water_level_m: float | None
    lower_m: float | None
    upper_m: float | None
    confidence: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    status: str = SUCCESS
    message: str = ""
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == SUCCESS and self.predicted_water_level_m is not None


class ForecastExpert(ABC):
    """Abstract base class for deterministic forecasting experts."""

    model_name: str
    spec: ClassVar[ExpertSpec]

    @abstractmethod
    def forecast(self, context: Any) -> ExpertForecast:
        """Return a forecast for ``context`` or an unavailable status."""

    def unavailable(self, context: Any, message: str) -> ExpertForecast:
        return ExpertForecast(
            model_name=self.model_name,
            forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc,
            horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=None,
            lower_m=None,
            upper_m=None,
            confidence=0.0,
            diagnostics={},
            status=UNAVAILABLE,
            message=message,
        )

    def failed(self, context: Any, message: str) -> ExpertForecast:
        return ExpertForecast(
            model_name=self.model_name,
            forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc,
            horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=None,
            lower_m=None,
            upper_m=None,
            confidence=0.0,
            diagnostics={},
            status=FAILED,
            message=message,
        )


def interval(prediction: float, half_width: float) -> tuple[float, float]:
    return float(prediction - half_width), float(prediction + half_width)


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
