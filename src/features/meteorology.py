"""Helpers for optional meteorological forcing covariates.

Wai does not ship a validated meteorological data product. These helpers make
the supported column contract explicit so future live evaluations can add wind,
pressure, rainfall, or wave covariates without changing the model API.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


METEOROLOGICAL_FORCING_COLUMNS: dict[str, str] = {
    "wind_speed_mps": "Wind speed in meters per second.",
    "wind_direction_deg": "Wind direction in degrees.",
    "air_pressure_hpa": "Air pressure in hectopascals.",
    "rainfall_mm": "Rainfall amount in millimeters for the sample interval.",
    "wave_height_m": "Significant wave height in meters.",
}


def supported_meteorological_columns() -> list[str]:
    """Return the stable column names recognized as external forcing inputs."""
    return list(METEOROLOGICAL_FORCING_COLUMNS)


def available_meteorological_columns(df: pd.DataFrame) -> list[str]:
    """Return supported forcing columns that are present in ``df``."""
    return [c for c in supported_meteorological_columns() if c in df.columns]


def audit_meteorological_columns(
    df: pd.DataFrame,
    required: Iterable[str] | None = None,
) -> dict:
    """Summarize whether a frame contains usable meteorological covariates."""
    required_cols = list(required or supported_meteorological_columns())
    present = [c for c in required_cols if c in df.columns]
    missing = [c for c in required_cols if c not in df.columns]
    non_numeric = [
        c for c in present if not pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric_present = [c for c in present if c not in non_numeric]
    complete_rows = int(df[numeric_present].dropna().shape[0]) if numeric_present else 0

    return {
        "supported_columns": supported_meteorological_columns(),
        "present_columns": present,
        "numeric_columns": numeric_present,
        "missing_columns": missing,
        "non_numeric_columns": non_numeric,
        "row_count": int(len(df)),
        "complete_forcing_rows": complete_rows,
        "usable": bool(numeric_present and not non_numeric),
    }
