"""Canonical observation schema helpers for regional-to-local forecasting.

The forecasting orchestrator consumes one schema regardless of source.  Source
adapters are responsible for translating provider-specific payloads into these
columns before any model or routing logic sees the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "timestamp_utc",
    "source",
    "station_id",
    "latitude",
    "longitude",
    "water_level_m",
    "datum",
    "record_type",
    "qc_status",
    "qc_flags",
    "retrieved_at",
    "latency_seconds",
]

ALLOWED_RECORD_TYPES = {
    "observation",
    "tide_prediction",
    "weather_observation",
    "forecast_guidance",
}

GOOD_QC_STATUSES = {"pass", "good", "verified", "preliminary", "unknown"}
BAD_QC_STATUSES = {"fail", "failed", "bad", "suspect", "rejected"}


class CanonicalSchemaError(ValueError):
    """Raised when data cannot be represented in the canonical schema."""


class DatumMismatchError(CanonicalSchemaError):
    """Raised when data on different vertical datums would be mixed."""


@dataclass(frozen=True)
class UnitConversion:
    """A small deterministic unit conversion descriptor."""

    source_unit: str
    multiplier_to_meters: float


UNIT_CONVERSIONS = {
    "m": UnitConversion("m", 1.0),
    "meter": UnitConversion("meter", 1.0),
    "meters": UnitConversion("meters", 1.0),
    "metre": UnitConversion("metre", 1.0),
    "metres": UnitConversion("metres", 1.0),
    "ft": UnitConversion("ft", 0.3048),
    "feet": UnitConversion("feet", 0.3048),
    "foot": UnitConversion("foot", 0.3048),
}


def utc_timestamp(value: object) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp from provider input."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def normalize_timestamp_series(values: pd.Series) -> pd.Series:
    """Normalize a Series of timestamps to pandas UTC datetimes."""

    return pd.to_datetime(values, utc=True)


def normalize_water_level_to_meters(values: pd.Series, units: object) -> pd.Series:
    """Convert a water-level Series into meters.

    ``units`` may be a scalar or a Series.  Mixed supported units are allowed
    only because each row is converted explicitly; unsupported units fail fast.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(units, pd.Series):
        out = []
        for value, unit in zip(numeric, units):
            key = str(unit).strip().lower()
            if key not in UNIT_CONVERSIONS:
                raise CanonicalSchemaError(f"Unsupported water-level unit: {unit!r}")
            out.append(value * UNIT_CONVERSIONS[key].multiplier_to_meters)
        return pd.Series(out, index=values.index, dtype="float64")

    key = str(units).strip().lower()
    if key not in UNIT_CONVERSIONS:
        raise CanonicalSchemaError(f"Unsupported water-level unit: {units!r}")
    return numeric * UNIT_CONVERSIONS[key].multiplier_to_meters


def canonicalize_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    record_type: str,
    timestamp_col: str = "timestamp",
    station_id_col: str = "station_id",
    water_level_col: str = "water_level",
    latitude_col: str = "lat",
    longitude_col: str = "lon",
    datum_col: str = "datum",
    units_col: str | None = "units",
    units: str = "m",
    qc_status_col: str | None = None,
    qc_flags_col: str | None = None,
    retrieved_at: object | None = None,
    extra_defaults: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Translate a provider frame into Wai's canonical observation schema."""

    if record_type not in ALLOWED_RECORD_TYPES:
        raise CanonicalSchemaError(f"Unsupported record_type: {record_type!r}")

    missing = [
        c for c in (
            timestamp_col,
            station_id_col,
            water_level_col,
            latitude_col,
            longitude_col,
            datum_col,
        )
        if c not in frame.columns
    ]
    if missing:
        raise CanonicalSchemaError(f"Missing required source columns: {missing}")

    df = frame.copy()
    retrieved = utc_timestamp(retrieved_at) if retrieved_at is not None else pd.Timestamp.now(tz="UTC")
    units_value = df[units_col] if units_col and units_col in df.columns else units

    out = pd.DataFrame({
        "timestamp_utc": normalize_timestamp_series(df[timestamp_col]),
        "source": str(source),
        "station_id": df[station_id_col].astype(str),
        "latitude": pd.to_numeric(df[latitude_col], errors="coerce"),
        "longitude": pd.to_numeric(df[longitude_col], errors="coerce"),
        "water_level_m": normalize_water_level_to_meters(df[water_level_col], units_value),
        "datum": df[datum_col].astype(str).str.upper(),
        "record_type": record_type,
        "retrieved_at": retrieved,
    })

    if qc_status_col and qc_status_col in df.columns:
        out["qc_status"] = df[qc_status_col].fillna("unknown").astype(str).str.lower()
    else:
        out["qc_status"] = "unknown"

    if qc_flags_col and qc_flags_col in df.columns:
        out["qc_flags"] = df[qc_flags_col].apply(_normalize_flags)
    else:
        out["qc_flags"] = [[] for _ in range(len(out))]

    if extra_defaults:
        for key, value in extra_defaults.items():
            out[key] = value

    retrieved_series = pd.Series([retrieved] * len(out), index=out.index)
    out["latency_seconds"] = (
        retrieved_series - out["timestamp_utc"]
    ).dt.total_seconds().astype(float)

    out = out[CANONICAL_COLUMNS]
    validate_canonical_observations(out)
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def validate_canonical_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a canonical observation frame.

    The function returns the input for ergonomic use in pipelines.
    """

    missing = [c for c in CANONICAL_COLUMNS if c not in frame.columns]
    if missing:
        raise CanonicalSchemaError(f"Canonical frame missing columns: {missing}")

    if frame.empty:
        return frame

    ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if ts.isna().any():
        raise CanonicalSchemaError("timestamp_utc contains invalid timestamps")

    retrieved = pd.to_datetime(frame["retrieved_at"], utc=True)
    if retrieved.isna().any():
        raise CanonicalSchemaError("retrieved_at contains invalid timestamps")

    invalid_types = sorted(set(frame["record_type"]) - ALLOWED_RECORD_TYPES)
    if invalid_types:
        raise CanonicalSchemaError(f"Invalid record_type values: {invalid_types}")

    if pd.to_numeric(frame["water_level_m"], errors="coerce").isna().any():
        raise CanonicalSchemaError("water_level_m contains non-numeric values")

    if frame["datum"].isna().any() or (frame["datum"].astype(str).str.strip() == "").any():
        raise CanonicalSchemaError("datum must be present for every record")

    return frame


def assert_compatible_datums(
    frames: Sequence[pd.DataFrame],
    *,
    label: str = "canonical frames",
) -> str:
    """Return the single datum in use or raise when datums differ.

    Wai intentionally does not perform vertical datum conversion yet.  Mixing
    MLLW, MSL, NAVD88, or provider-specific datums without a verified
    conversion would silently corrupt residual forecasts, so this check fails
    closed.
    """

    datums: set[str] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if "datum" not in frame.columns:
            raise DatumMismatchError(f"{label}: missing datum column")
        datums.update(str(v).upper() for v in frame["datum"].dropna().unique())
    if not datums:
        raise DatumMismatchError(f"{label}: no datum values available")
    if len(datums) > 1:
        raise DatumMismatchError(f"{label}: incompatible datums {sorted(datums)}")
    return next(iter(datums))


def is_good_qc(status: object) -> bool:
    """Return whether a QC status is usable for local forecasting."""

    value = str(status or "unknown").lower()
    return value in GOOD_QC_STATUSES and value not in BAD_QC_STATUSES


def _normalize_flags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [part.strip() for part in stripped.split(",") if part.strip()]
    return [str(value)]
