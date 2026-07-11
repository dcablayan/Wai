"""NOAA station data helpers used by the refactored pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def _normalize_station_id(station_id):
    """Normalize station ids like 9410230 / 9410230.0 / '9410230'."""
    sid = str(station_id).strip()
    if sid.endswith(".0"):
        sid = sid[:-2]
    if sid.isdigit():
        return sid
    try:
        return str(int(float(sid)))
    except Exception:
        return sid


def get_noaa_paths(station_id):
    root = Path("./data")
    sid = _normalize_station_id(station_id)
    candidates: List[Path] = [
        root / f"{sid}.csv",
        root / f"{sid}.tsv",
        root / f"noaa_{sid}.csv",
        root / f"noaa_{sid}.tsv",
        root / "noaa" / f"{sid}.csv",
        root / "noaa" / f"{sid}.tsv",
    ]
    return [p for p in candidates if p.exists()]


def _to_datetime_safe(values):
    out = pd.to_datetime(values, errors="coerce", utc=True)
    if out.notna().all():
        return out
    return pd.to_datetime(values, errors="coerce")


def _read_noaa_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    cols = [c.lower() for c in df.columns]
    time_col = next((name for name in df.columns if name.lower() in {"time", "timestamp", "datetime"}), None)
    if time_col is None:
        first_col = df.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df[first_col]):
            time_col = first_col
        else:
            raise ValueError(f"Could not detect time column in {path}")

    value_candidates = [
        "water_level",
        "waterlevel",
        "water_level_m",
        "value",
        "wl",
    ]
    value_col = next((c for c in df.columns if c.lower() in value_candidates), None)
    if value_col is None:
        # Fallback: take first non-time column.
        remaining = [c for c in df.columns if c != time_col]
        if not remaining:
            raise ValueError(f"Could not detect value column in {path}")
        value_col = remaining[0]

    df = df[[time_col, value_col]].copy()
    df[time_col] = _to_datetime_safe(df[time_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df.sort_values(time_col).set_index(time_col)


def fetch_noaa_data(noaa_id, product, begin, end, units="metric"):
    paths = get_noaa_paths(noaa_id)
    if not paths:
        raise FileNotFoundError(f"No local NOAA file found for station {noaa_id}")
    sid = _normalize_station_id(noaa_id)

    raw = _read_noaa_file(paths[0])
    value_col = raw.columns[0]
    raw = raw.rename(columns={value_col: sid}).dropna(how="all")

    begin_ts = pd.to_datetime(begin, unit="s", utc=True)
    end_ts = pd.to_datetime(end, unit="s", utc=True)
    raw = raw[(raw.index >= begin_ts) & (raw.index <= end_ts)]

    if units == "english":
        raw = raw * 3.28084
    elif units != "metric":
        raise ValueError("units must be 'metric' or 'english'")

    return raw
