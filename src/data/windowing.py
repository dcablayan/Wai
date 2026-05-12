"""Time-series windowing utilities for Wai.

Core logic adapted from dcablayan/tideformer data_utils.py:
  make_windows(), train_val_test_split()

Extended with DataFrame support and a loader for the tidecast CSV format.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Type alias used throughout this module
Window = Dict[str, object]


# ── Window construction ───────────────────────────────────────────────────────

def make_windows(
    series: List[float],
    lookback: int,
    horizon: int = 1,
    max_samples: Optional[int] = None,
    times: Optional[List[float]] = None,
) -> List[Window]:
    """Convert a flat value list into supervised sliding windows.

    Each window dict carries:
      values       : list[float]  — lookback observations
      times        : list[float]  — corresponding fractional hours
      target_value : float        — next-step value (horizon=1)
      target_time  : float        — timestamp of target

    Parameters
    ----------
    series : list[float]
        Ordered water-level values.
    lookback : int
        Number of past steps to include as context.
    horizon : int
        Steps ahead for the target (default 1).
    max_samples : int, optional
        Cap number of windows created (useful during benchmarking).
    times : list[float], optional
        Fractional hours aligned with series. Auto-generated (0,1,2,…)
        if not provided.
    """
    if times is None:
        times = [float(i) for i in range(len(series))]

    n_windows = len(series) - lookback - horizon + 1
    if n_windows <= 0:
        return []
    if max_samples is not None:
        n_windows = min(n_windows, max_samples)

    windows: List[Window] = []
    for i in range(n_windows):
        windows.append({
            "values": series[i: i + lookback],
            "times": times[i: i + lookback],
            "target_value": series[i + lookback + horizon - 1],
            "target_time": times[i + lookback + horizon - 1],
        })
    return windows


def temporal_split(
    windows: List[Window],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[List[Window], List[Window], List[Window]]:
    """Split windows into train / val / test by time order (no shuffling).

    Default 70 / 15 / 15 split matches dcablayan/tideformer convention.
    """
    n = len(windows)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return windows[:train_end], windows[train_end:val_end], windows[val_end:]


# ── Tidecast loader ───────────────────────────────────────────────────────────

def load_tidecast_series(path: str | Path) -> Tuple[List[float], List[float]]:
    """Load a hohonu-*_tidecast.csv into (times_hours, values) lists.

    The tidecast format from dcablayan/tideformer has two columns:
      dt         — ISO 8601 UTC timestamp
      prediction — water-level prediction (feet, MLLW)

    Times are returned as fractional hours from the first observation
    so they can be used directly with harmonic computations.
    """
    path = Path(path)
    times: List[float] = []
    values: List[float] = []
    base_time: Optional[datetime] = None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_raw = row.get("dt", "").strip()
            val_raw = row.get("prediction", "").strip()
            if not dt_raw or not val_raw:
                continue
            try:
                dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
                value = float(val_raw)
            except (ValueError, TypeError):
                continue
            if base_time is None:
                base_time = dt
            hours = (dt - base_time).total_seconds() / 3600.0
            times.append(hours)
            values.append(value)

    return times, values


def load_tidecast_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a tidecast CSV into a Wai-schema DataFrame.

    The returned DataFrame conforms to the standard schema:
    timestamp, station_id, water_level, datum, units, lat, lon, source.

    Station coordinates are not embedded in the tidecast files; lat/lon
    are left as NaN. Units are 'ft' per NOAA CO-OPS convention for
    tidal predictions. Source is set to 'TIDECAST'.
    """
    path = Path(path)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_raw = row.get("dt", "").strip()
            val_raw = row.get("prediction", "").strip()
            if not dt_raw or not val_raw:
                continue
            try:
                dt = pd.Timestamp(dt_raw).tz_convert("UTC")
                value = float(val_raw)
            except Exception:
                continue
            rows.append({
                "timestamp": dt,
                "water_level": value,
            })

    if not rows:
        raise ValueError(f"No valid rows in {path.name}")

    station_id = path.stem.replace("_tidecast", "")
    df = pd.DataFrame(rows)
    df["station_id"] = station_id
    df["datum"] = "MLLW"
    df["units"] = "ft"
    df["lat"] = float("nan")
    df["lon"] = float("nan")
    df["source"] = "TIDECAST"
    return df.sort_values("timestamp").reset_index(drop=True)
