"""Cadence inference, despiking, and resampling for arbitrary gauge data.

Wai's lag and rolling features operate on row position, so they are only
correct on a regular time grid.  Provider gauges sample anywhere from 1 to 15
minutes, drop records during outages, and occasionally emit spikes.  This
module turns any single-station observation series into a regular, QC'd grid
before feature engineering sees it.

All functions accept either vocabulary (canonical ``timestamp_utc`` /
``water_level_m`` or model ``timestamp`` / ``water_level``) by autodetecting
column names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _detect_columns(frame: pd.DataFrame) -> tuple[str, str]:
    if "timestamp_utc" in frame.columns and "water_level_m" in frame.columns:
        return "timestamp_utc", "water_level_m"
    if "timestamp" in frame.columns and "water_level" in frame.columns:
        return "timestamp", "water_level"
    raise ValueError(
        "Frame has neither canonical (timestamp_utc/water_level_m) nor model "
        f"(timestamp/water_level) columns; got {list(frame.columns)}"
    )


def infer_cadence_minutes(timestamps: pd.Series) -> float:
    """Return the modal sampling interval in minutes.

    Uses the mode of consecutive deltas so isolated gaps and duplicates do not
    skew the estimate the way a mean would.
    """

    ts = pd.to_datetime(timestamps, utc=True).sort_values()
    deltas = ts.diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[deltas > 0]
    if deltas.empty:
        raise ValueError("Need at least two distinct timestamps to infer cadence")
    return float(deltas.mode().iloc[0])


def despike_mad(
    values: pd.Series,
    *,
    window: int = 21,
    threshold: float = 6.0,
) -> pd.Series:
    """Return a boolean mask of spike rows via a rolling-median MAD test.

    A point is a spike when it deviates from the centered rolling median by
    more than ``threshold`` scaled median-absolute-deviations.  The default
    threshold is deliberately loose: tides swing fast, and a false positive
    deletes real signal while a false negative only adds noise.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    med = numeric.rolling(window, center=True, min_periods=3).median()
    abs_dev = (numeric - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=3).median()
    # 1.4826 scales MAD to a normal-consistent sigma; floor keeps flat calm
    # water (MAD ~ 0) from flagging millimeter sensor noise as spikes.
    sigma = (1.4826 * mad).clip(lower=0.02)
    return (abs_dev > threshold * sigma).fillna(False)


@dataclass(frozen=True)
class RegularizeReport:
    """What ``regularize_frame`` did to the series."""

    cadence_minutes: float
    n_input: int
    n_output: int
    n_spikes_removed: int
    n_interpolated: int
    n_gap_rows: int


def regularize_frame(
    frame: pd.DataFrame,
    *,
    cadence_minutes: float | None = None,
    max_gap_minutes: float | None = None,
    despike: bool = True,
    despike_threshold: float = 6.0,
) -> tuple[pd.DataFrame, RegularizeReport]:
    """Return (regular-grid frame, report) for a single-station series.

    - Infers cadence from the data when not supplied.
    - Removes MAD spikes (optional), then snaps records to a regular grid.
    - Linearly interpolates gaps up to ``max_gap_minutes`` (default: 3
      cadences); longer gaps stay NaN so downstream feature building drops
      them instead of inventing water levels across an outage.
    - Adds an ``is_interpolated`` boolean column.

    Non-time metadata columns (station_id, datum, source, ...) are forward
    filled; extra numeric covariate columns are interpolated under the same
    gap policy as the water level.
    """

    ts_col, wl_col = _detect_columns(frame)
    if frame.empty:
        raise ValueError("Cannot regularize an empty frame")

    df = frame.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df[wl_col] = pd.to_numeric(df[wl_col], errors="coerce")
    df = df.sort_values(ts_col).drop_duplicates(subset=ts_col, keep="last")
    n_input = len(df)

    if cadence_minutes is None:
        cadence_minutes = infer_cadence_minutes(df[ts_col])
    if cadence_minutes <= 0:
        raise ValueError(f"cadence_minutes must be positive, got {cadence_minutes}")
    if max_gap_minutes is None:
        max_gap_minutes = 3.0 * cadence_minutes

    n_spikes = 0
    if despike and len(df) >= 5:
        spikes = despike_mad(df[wl_col], threshold=despike_threshold)
        n_spikes = int(spikes.sum())
        df.loc[spikes, wl_col] = np.nan

    freq = pd.Timedelta(minutes=cadence_minutes)
    grid_start = df[ts_col].iloc[0].ceil(freq)
    grid_end = df[ts_col].iloc[-1].floor(freq)
    if grid_end < grid_start:
        grid_start = df[ts_col].iloc[0]
        grid_end = df[ts_col].iloc[-1]
    grid = pd.date_range(grid_start, grid_end, freq=freq, tz="UTC")

    indexed = df.set_index(ts_col)
    numeric_cols = [
        c for c in indexed.columns if pd.api.types.is_numeric_dtype(indexed[c])
    ]
    meta_cols = [c for c in indexed.columns if c not in numeric_cols]

    # Snap each grid point to the nearest record within half a cadence, so
    # slightly jittered timestamps (e.g. 12:00:03) land on the grid instead of
    # being interpolated.
    tolerance = freq / 2
    on_grid = indexed[numeric_cols].reindex(grid, method="nearest", tolerance=tolerance)

    limit = max(1, int(round(max_gap_minutes / cadence_minutes)) - 1)
    was_nan = on_grid[wl_col].isna()
    filled = on_grid.interpolate(method="time", limit=limit, limit_area="inside")
    is_interpolated = was_nan & filled[wl_col].notna()

    out = filled.reset_index().rename(columns={"index": ts_col})
    out["is_interpolated"] = is_interpolated.to_numpy()

    if meta_cols:
        meta = indexed[meta_cols].reindex(grid, method="ffill").bfill()
        for col in meta_cols:
            out[col] = meta[col].to_numpy()

    n_gap_rows = int(out[wl_col].isna().sum())
    report = RegularizeReport(
        cadence_minutes=float(cadence_minutes),
        n_input=n_input,
        n_output=len(out),
        n_spikes_removed=n_spikes,
        n_interpolated=int(is_interpolated.sum()),
        n_gap_rows=n_gap_rows,
    )
    return out, report


def steps_for_minutes(minutes: float, cadence_minutes: float) -> int:
    """Convert a physical duration into a whole number of grid steps."""

    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive")
    return max(1, int(round(minutes / cadence_minutes)))
