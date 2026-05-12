"""Tidal feature engineering for water-level forecasting."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Major tidal constituents with their periods in hours
TIDAL_CONSTITUENTS = {
    "M2": 12.4206,   # principal lunar semi-diurnal
    "S2": 12.0000,   # principal solar semi-diurnal
    "K1": 23.9345,   # lunisolar diurnal
    "O1": 25.8193,   # principal lunar diurnal
    "N2": 12.6583,   # larger lunar elliptic semi-diurnal
}

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


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    value_col: str = "water_level",
) -> pd.DataFrame:
    """Add lagged water-level features (in 6-min steps by default)."""
    df = df.copy()
    if lags is None:
        lags = [1, 2, 4, 10, 20, 40]  # ~6min, 12min, 24min, 1hr, 2hr, 4hr
    for lag in lags:
        df[f"{value_col}_lag{lag}"] = df[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    value_col: str = "water_level",
) -> pd.DataFrame:
    """Add rolling mean and std features."""
    df = df.copy()
    if windows is None:
        windows = [10, 40, 240]  # ~1hr, 4hr, 24hr at 6-min resolution
    for w in windows:
        df[f"{value_col}_rmean{w}"] = df[value_col].rolling(w, min_periods=1).mean()
        df[f"{value_col}_rstd{w}"] = (
            df[value_col].rolling(w, min_periods=2).std().fillna(0.0)
        )
    return df


NON_FEATURE_COLS = {
    "timestamp", "station_id", "datum", "units", "lat", "lon", "source",
}


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "water_level",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) from a single-station time series.

    Applies tidal harmonics, lag features, and rolling statistics,
    then drops rows with any NaN from the lag computation.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df = add_tidal_harmonics(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and c != target_col
    ]
    return df[feature_cols], df[target_col]
