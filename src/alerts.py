"""High-water alert detection for Wai.

Three configurable threshold modes
-----------------------------------
'std'        mean + k * std_dev  (default k=2.0, recommended starting point)
'absolute'   fixed water-level value in the same units as the series
'percentile' p-th percentile of a reference distribution (default p=95)

Usage
-----
    from src.alerts import AlertConfig, detect_alerts, generate_alert_summary

    config = AlertConfig(mode="std", k=2.0)
    alert_df = detect_alerts(df, config)
    summary = generate_alert_summary(df, config, station_id="DEMO-HNL")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


AlertMode = Literal["std", "absolute", "percentile"]


@dataclass
class AlertConfig:
    mode: AlertMode = "std"
    k: float = 2.0
    absolute_threshold: Optional[float] = None
    percentile: float = 95.0


def compute_threshold(series: pd.Series, config: AlertConfig) -> float:
    """Derive the scalar alert threshold from the config and reference series."""
    vals = series.dropna()
    if len(vals) == 0:
        return float("nan")

    if config.mode == "std":
        return float(vals.mean() + config.k * vals.std())
    elif config.mode == "absolute":
        if config.absolute_threshold is None:
            raise ValueError(
                "absolute_threshold must be set when mode='absolute'"
            )
        return float(config.absolute_threshold)
    elif config.mode == "percentile":
        return float(np.percentile(vals, config.percentile))
    else:
        raise ValueError(f"Unknown alert mode: {config.mode!r}")


def detect_alerts(
    df: pd.DataFrame,
    config: AlertConfig,
    value_col: str = "water_level",
    reference_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return rows in df where water_level exceeds the computed threshold.

    Parameters
    ----------
    df : DataFrame
        Must contain value_col.
    config : AlertConfig
        Threshold configuration.
    value_col : str
        Column name for water-level values.
    reference_series : pd.Series, optional
        Series used to fit the threshold statistics (e.g., training data).
        If None, the threshold is computed from df[value_col].

    Returns
    -------
    DataFrame of alert rows with added 'threshold' and 'alert_mode' columns.
    """
    ref = reference_series if reference_series is not None else df[value_col]
    threshold = compute_threshold(ref, config)
    alerts = df[df[value_col] >= threshold].copy()
    alerts["threshold"] = threshold
    alerts["alert_mode"] = config.mode
    return alerts


def generate_alert_summary(
    df: pd.DataFrame,
    config: AlertConfig,
    station_id: str,
    value_col: str = "water_level",
    reference_series: Optional[pd.Series] = None,
) -> dict:
    """Return a summary dict suitable for JSON serialisation and HTML reports.

    Parameters
    ----------
    df : DataFrame with value_col and (optionally) 'timestamp'.
    config : AlertConfig
    station_id : str
    value_col : str
    reference_series : pd.Series, optional

    Returns
    -------
    dict with keys: station_id, alert_mode, threshold, n_total_obs,
    n_alerts, alert_rate_pct, events (list of dicts).
    """
    ref = reference_series if reference_series is not None else df[value_col]
    threshold = compute_threshold(ref, config)
    alerts = detect_alerts(df, config, value_col, reference_series)

    events = []
    for _, row in alerts.head(50).iterrows():
        ts = row.get("timestamp")
        if pd.notna(ts) and hasattr(ts, "strftime"):
            ts_str = ts.strftime("%Y-%m-%d %H:%M UTC")
        else:
            ts_str = str(ts) if ts is not None else "—"
        events.append({
            "timestamp": ts_str,
            "value": round(float(row[value_col]), 4),
            "threshold": round(float(threshold), 4),
        })

    return {
        "station_id": station_id,
        "alert_mode": config.mode,
        "threshold": round(float(threshold), 4),
        "n_total_obs": int(len(df)),
        "n_alerts": int(len(alerts)),
        "alert_rate_pct": round(100.0 * len(alerts) / max(len(df), 1), 2),
        "events": events,
    }


def save_alert_summary(summary: dict, path: Path) -> None:
    """Write alert summary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
