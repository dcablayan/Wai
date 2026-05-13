"""High-water alert detection for Wai.

Three configurable threshold modes
-----------------------------------
'std'        mean + k * std_dev  (default k=2.0, recommended starting point)
'absolute'   fixed water-level value in the same units as the series
'percentile' p-th percentile of a reference distribution (default p=95)

Threshold fitting
-----------------
Always pass a *reference_series* (e.g. the training split) to fit the threshold
statistics.  Fitting on the currently displayed/filtered range produces a
threshold that tracks the display window rather than the climatological baseline.

Usage
-----
    from src.alerts import AlertConfig, detect_alerts, group_alert_episodes

    config = AlertConfig(mode="std", k=2.0)
    alert_df = detect_alerts(df, config, reference_series=train["water_level"])
    episodes = group_alert_episodes(alert_df, threshold=config_threshold)
    summary = generate_alert_summary(df, config, station_id="DEMO-HNL",
                                     reference_series=train["water_level"])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

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
        Series used to fit the threshold statistics (training data recommended).
        If None, the threshold is computed from df[value_col] — this conflates
        the reference distribution with the evaluation window and should be
        avoided for production use.

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


def group_alert_episodes(
    alerts_df: pd.DataFrame,
    threshold: float,
    value_col: str = "water_level",
    timestamp_col: str = "timestamp",
    gap_steps: int = 1,
) -> List[dict]:
    """Group consecutive alert rows into episodes.

    Two alert rows are considered part of the same episode if they are
    separated by at most *gap_steps* missing rows in the original index.
    This converts sample-level exceedances into contiguous high-water events.

    Parameters
    ----------
    alerts_df : pd.DataFrame
        Output of detect_alerts — rows where water_level >= threshold.
    threshold : float
        The threshold value used to detect alerts (stored in each episode dict).
    value_col : str
        Column carrying water-level values.
    timestamp_col : str
        Column carrying timestamps (optional; episodes include it when present).
    gap_steps : int
        Maximum gap in integer index positions between two rows that are still
        considered the same episode.  Default = 1 (consecutive rows only).

    Returns
    -------
    List of episode dicts, each with:
        start           — timestamp or row index of first alert sample
        end             — timestamp or row index of last alert sample
        duration_steps  — number of contiguous alert samples
        peak            — maximum water_level value in the episode
        exceedance_m    — peak minus threshold (in the same units as water_level)
    """
    if alerts_df.empty:
        return []

    df = alerts_df.reset_index(drop=False).copy()
    if "index" not in df.columns:
        df["_row"] = np.arange(len(df))
        idx_col = "_row"
    else:
        idx_col = "index"

    episodes: List[dict] = []
    group_rows = [df.iloc[0]]

    for i in range(1, len(df)):
        prev_idx = group_rows[-1][idx_col]
        curr_idx = df.iloc[i][idx_col]
        if curr_idx - prev_idx <= gap_steps:
            group_rows.append(df.iloc[i])
        else:
            episodes.append(_episode_from_rows(group_rows, threshold, value_col, timestamp_col))
            group_rows = [df.iloc[i]]

    episodes.append(_episode_from_rows(group_rows, threshold, value_col, timestamp_col))
    return episodes


def _episode_from_rows(rows: list, threshold: float, value_col: str, timestamp_col: str) -> dict:
    values = [float(r[value_col]) for r in rows]
    peak = max(values)

    def _fmt_ts(r):
        ts = r.get(timestamp_col)
        if ts is not None and pd.notna(ts) and hasattr(ts, "strftime"):
            return ts.strftime("%Y-%m-%d %H:%M UTC")
        return str(ts) if ts is not None else "—"

    return {
        "start": _fmt_ts(rows[0]),
        "end": _fmt_ts(rows[-1]),
        "duration_steps": len(rows),
        "peak": round(peak, 4),
        "exceedance_m": round(peak - threshold, 4),
    }


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
        Strongly recommended: pass the training split so the threshold is fit
        on historical/climatological data, not the currently displayed range.

    Returns
    -------
    dict with keys: station_id, alert_mode, threshold, n_total_obs,
    n_alerts, n_episodes, alert_rate_pct, episodes (list of episode dicts).
    """
    ref = reference_series if reference_series is not None else df[value_col]
    threshold = compute_threshold(ref, config)
    alerts = detect_alerts(df, config, value_col, reference_series)
    episodes = group_alert_episodes(alerts, threshold=threshold, value_col=value_col)

    return {
        "station_id": station_id,
        "alert_mode": config.mode,
        "threshold": round(float(threshold), 4),
        "n_total_obs": int(len(df)),
        "n_alerts": int(len(alerts)),
        "n_episodes": int(len(episodes)),
        "alert_rate_pct": round(100.0 * len(alerts) / max(len(df), 1), 2),
        "episodes": episodes[:50],
    }


def save_alert_summary(summary: dict, path: Path) -> None:
    """Write alert summary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
