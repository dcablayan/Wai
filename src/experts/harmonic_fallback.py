"""Harmonic regression expert fit on local observations only.

Every other tide-informed expert needs an externally supplied tide-prediction
product (NOAA or local harmonics file).  Gauges outside the NOAA network have
no such product, which used to leave persistence as the only real expert.
This expert closes that gap: it least-squares fits sin/cos amplitudes for the
major tidal constituents directly to the gauge's own recent history, so any
station with a couple of days of data gets a tide-shaped forecast.

The fit happens per-forecast on a bounded lookback window, which keeps the
expert deterministic and leakage-safe (only rows at or before the forecast
origin are visible via the prepared slices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experts.base import ExpertForecast, ForecastExpert, clamp_confidence, interval
from src.experts.capabilities import LATENCY_MODERATE, ExpertSpec
from src.features.engineering import EPOCH, TIDAL_CONSTITUENTS

# Constituents ordered by typical amplitude; a constituent is used only when
# the training window spans enough of its cycles to resolve it.
_MIN_CYCLES = 1.5
_MIN_SPAN_HOURS = 48.0
_MIN_SAMPLES = 100
_LOOKBACK_DAYS = 14.0


class HarmonicFallbackExpert(ForecastExpert):
    """Tide forecast from constituents fit to the local gauge itself."""

    model_name = "harmonic_fallback"
    spec = ExpertSpec(
        model_name="harmonic_fallback",
        required_sources=("hohonu_observation",),
        requires_local_obs=True,
        requires_tide=False,
        min_horizon_minutes=0,
        max_horizon_minutes=7 * 24 * 60,
        latency_class=LATENCY_MODERATE,
        compute_cost=4.0,
        notes=(
            "Least-squares harmonic fit on the gauge's own history; the only "
            "tide-shaped expert available when no tide-prediction product exists."
        ),
    )

    def forecast(self, context) -> ExpertForecast:
        history = self._local_history(context)
        if history is None or len(history) < _MIN_SAMPLES:
            return self.unavailable(
                context,
                f"needs at least {_MIN_SAMPLES} local observations to fit harmonics",
            )
        if not context.hohonu_qc_ok:
            return self.unavailable(context, "latest local observation failed QC")

        timestamps = pd.to_datetime(history["timestamp_utc"], utc=True)
        span_hours = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600.0
        if span_hours < _MIN_SPAN_HOURS:
            return self.unavailable(
                context,
                f"needs {_MIN_SPAN_HOURS:.0f}h of local history, have {span_hours:.1f}h",
            )

        constituents = {
            name: period
            for name, period in TIDAL_CONSTITUENTS.items()
            if span_hours >= _MIN_CYCLES * period
        }
        if not constituents:
            return self.unavailable(context, "window too short for any constituent")

        t_hours = (timestamps - EPOCH).dt.total_seconds().to_numpy() / 3600.0
        levels = pd.to_numeric(history["water_level_m"], errors="coerce").to_numpy()
        valid = np.isfinite(levels)
        if valid.sum() < _MIN_SAMPLES:
            return self.unavailable(context, "too few finite local observations")
        t_hours, levels = t_hours[valid], levels[valid]

        design = _design_matrix(t_hours, constituents)
        coef, *_ = np.linalg.lstsq(design, levels, rcond=None)
        fitted = design @ coef
        residual_std = float(np.std(levels - fitted))

        target_hours = (context.target_time_utc - EPOCH).total_seconds() / 3600.0
        prediction = (
            _design_matrix(np.array([target_hours]), constituents) @ coef
        ).item()

        horizon_hours = context.horizon_minutes / 60.0
        half_width = max(0.05, residual_std * (1.5 + 0.05 * horizon_hours))
        lower, upper = interval(prediction, half_width)
        confidence = clamp_confidence(
            0.6 - 0.01 * horizon_hours - min(residual_std, 0.3)
        )
        return ExpertForecast(
            model_name=self.model_name,
            forecast_time_utc=context.forecast_time_utc,
            target_time_utc=context.target_time_utc,
            horizon_minutes=context.horizon_minutes,
            predicted_water_level_m=prediction,
            lower_m=lower,
            upper_m=upper,
            confidence=confidence,
            diagnostics={
                "constituents": sorted(constituents),
                "train_samples": int(len(levels)),
                "train_span_hours": float(span_hours),
                "residual_std_m": residual_std,
            },
        )

    def _local_history(self, context) -> pd.DataFrame | None:
        prepared = getattr(context, "prepared", None)
        if prepared is not None and len(prepared.hohonu):
            start = context.forecast_time_utc - pd.Timedelta(days=_LOOKBACK_DAYS)
            return prepared.recent_slice(
                prepared.hohonu,
                prepared.hohonu_ts,
                start,
                context.forecast_time_utc,
            )
        recent = context.recent_hohonu_observations
        if recent is not None and len(recent):
            return recent
        return None


def _design_matrix(t_hours: np.ndarray, constituents: dict[str, float]) -> np.ndarray:
    columns = [np.ones_like(t_hours)]
    for period in constituents.values():
        omega = 2.0 * np.pi / period
        columns.append(np.sin(omega * t_hours))
        columns.append(np.cos(omega * t_hours))
    return np.column_stack(columns)
