"""Tidal feature engineering for water-level forecasting."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Eight tidal constituents with their periods in hours.
# Five core (M2, S2, K1, O1, N2) + three added from HarmonicNetPrototype
# (dcablayan/tideformer): shallow-water M4/M6 and long-period Mm.
TIDAL_CONSTITUENTS = {
    "M2": 12.4206,    # principal lunar semi-diurnal
    "S2": 12.0000,    # principal solar semi-diurnal
    "K1": 23.9345,    # lunisolar diurnal
    "O1": 25.8193,    # principal lunar diurnal
    "N2": 12.6583,    # larger lunar elliptic semi-diurnal
    "M4": 6.2103,     # shallow-water overtide of M2
    "M6": 4.1402,     # shallow-water overtide of M2 (3rd harmonic)
    "Mm": 327.8599,   # lunar monthly (27.32 days)
}

# Synodic (full-cycle) lunar month in hours — for lunar-phase feature
_LUNAR_SYNODIC_HOURS = 29.53 * 24.0

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _hours_since_epoch(timestamps: pd.Series) -> np.ndarray:
    return (timestamps - EPOCH).dt.total_seconds().values / 3600.0


def add_tidal_harmonics(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    constituents: Optional[dict] = None,
) -> pd.DataFrame:
    """Add sin/cos features for major tidal constituents."""
    df = df.copy()
    constits = constituents or TIDAL_CONSTITUENTS
    t_hours = _hours_since_epoch(df[timestamp_col])
    for name, period_h in constits.items():
        omega = 2 * np.pi / period_h
        df[f"tide_sin_{name}"] = np.sin(omega * t_hours)
        df[f"tide_cos_{name}"] = np.cos(omega * t_hours)
    return df


def add_temporal_covariates(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add hour-of-day and lunar-phase sin/cos features.

    Adapted from hour_of_day() and lunar_phase() helpers in
    dcablayan/tideformer prototypes.py.
    """
    df = df.copy()
    t_hours = _hours_since_epoch(df[timestamp_col])
    hod = (t_hours % 24.0) / 24.0
    df["hour_sin"] = np.sin(2 * np.pi * hod)
    df["hour_cos"] = np.cos(2 * np.pi * hod)
    lp = (t_hours % _LUNAR_SYNODIC_HOURS) / _LUNAR_SYNODIC_HOURS
    df["lunar_sin"] = np.sin(2 * np.pi * lp)
    df["lunar_cos"] = np.cos(2 * np.pi * lp)
    return df


# Physical durations behind the historical step-count defaults, which assumed
# the 6-minute NOAA cadence. Cadence-aware callers pass ``cadence_minutes`` so
# these same durations hold on any regular grid.
LAG_MINUTES = [6.0, 12.0, 24.0, 60.0, 120.0, 240.0]
WINDOW_MINUTES = [60.0, 240.0, 1440.0]
DEFAULT_CADENCE_MINUTES = 6.0


def _steps(minutes_list: List[float], cadence_minutes: float) -> List[int]:
    """Convert physical durations to whole grid steps, deduplicated in order."""
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive")
    steps: List[int] = []
    for minutes in minutes_list:
        step = max(1, int(round(minutes / cadence_minutes)))
        if step not in steps:
            steps.append(step)
    return steps


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    value_col: str = "water_level",
    cadence_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Add lagged water-level features.

    ``lags`` are grid steps. When omitted they cover ``LAG_MINUTES`` at the
    given cadence (default 6-minute grid, matching the historical
    ``[1, 2, 4, 10, 20, 40]`` steps). The series must already be on a regular
    grid — see ``src.data.regularize`` for irregular provider data.
    """
    df = df.copy()
    if lags is None:
        lags = _steps(LAG_MINUTES, cadence_minutes or DEFAULT_CADENCE_MINUTES)
    for lag in lags:
        df[f"{value_col}_lag{lag}"] = df[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    value_col: str = "water_level",
    cadence_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Add rolling mean and std features.

    The series is shifted by 1 before windowing so that the feature at row t
    is computed from [t-w, t-1], never including water_level[t] itself.
    This prevents target leakage when the model predicts water_level[t].
    """
    df = df.copy()
    if windows is None:
        windows = _steps(WINDOW_MINUTES, cadence_minutes or DEFAULT_CADENCE_MINUTES)
    shifted = df[value_col].shift(1)
    for w in windows:
        df[f"{value_col}_rmean{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"{value_col}_rstd{w}"] = shifted.rolling(w, min_periods=2).std().fillna(0.0)
    return df


NON_FEATURE_COLS = {
    "timestamp", "station_id", "datum", "units", "lat", "lon", "source",
    "_source_row", "observed_water_level", "observation_source",
    "_target_h", "noaa_prediction", "prediction_source",
    "is_interpolated", "latency_seconds",
}


def feature_columns(df: pd.DataFrame, target_col: str = "water_level") -> list[str]:
    """Return numeric, non-target columns used by tabular models.

    The matrix may include externally supplied meteorological covariates such
    as wind speed or pressure when they are present and numeric. Known NOAA
    comparison fields stay excluded so a baseline column cannot leak into the
    generic harmonic model by accident.
    """
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c != target_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_feature_frame(
    df: pd.DataFrame,
    target_col: str = "water_level",
    cadence_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Build an aligned feature DataFrame from a single-station time series.

    Applies tidal harmonics (8 constituents), temporal covariates
    (hour-of-day, lunar phase), lag features, and rolling statistics,
    then drops rows with any NaN from the lag computation. The returned frame
    keeps ``timestamp``, the target column, and ``_source_row`` so callers can
    align predictions, observations, and plot timestamps without rebuilding the
    feature matrix on sliced data.

    ``cadence_minutes`` is the series' regular sampling interval; lag and
    rolling windows are sized so they always span the same physical durations.
    Omitting it keeps the historical 6-minute assumption.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["_source_row"] = np.arange(len(df), dtype=int)
    df = add_tidal_harmonics(df)
    df = add_temporal_covariates(df)
    df = add_lag_features(df, cadence_minutes=cadence_minutes)
    df = add_rolling_features(df, cadence_minutes=cadence_minutes)
    # Drop rows only for NaNs that reach the model (lag warm-up, data gaps).
    # NaN metadata such as missing lat/lon must not erase valid samples.
    subset = feature_columns(df, target_col=target_col) + [target_col]
    return df.dropna(subset=subset).reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "water_level",
    cadence_minutes: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) from a single-station time series."""
    df = build_feature_frame(df, target_col=target_col, cadence_minutes=cadence_minutes)

    feature_cols = feature_columns(df, target_col=target_col)
    return df[feature_cols], df[target_col]
