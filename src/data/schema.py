"""Bridge between Wai's two observation vocabularies.

Historically the repo grew two column vocabularies:

- The **canonical** schema (``src/data/canonicalize.py``): ``timestamp_utc``,
  ``water_level_m``, ``latitude``/``longitude`` plus provenance and QC columns.
  Emitted by source adapters and consumed by the forecast orchestrator.
- The **model** schema (``src/data/loader.py``, ``src/data/validation.py``):
  ``timestamp``, ``water_level``, ``lat``/``lon``, ``units``.  Consumed by
  feature engineering, the tabular models, validation, and the dashboard.

This module is the single place that maps between them so a frame from any
source adapter can feed the tabular model path and vice versa.
"""

from __future__ import annotations

import pandas as pd

from src.data.canonicalize import (
    CANONICAL_COLUMNS,
    canonicalize_frame,
)

# canonical column -> model column
CANONICAL_TO_MODEL = {
    "timestamp_utc": "timestamp",
    "water_level_m": "water_level",
    "latitude": "lat",
    "longitude": "lon",
}

MODEL_TO_CANONICAL = {v: k for k, v in CANONICAL_TO_MODEL.items()}

MODEL_REQUIRED_COLUMNS = [
    "timestamp",
    "station_id",
    "water_level",
    "datum",
    "units",
    "lat",
    "lon",
    "source",
]


def is_canonical(frame: pd.DataFrame) -> bool:
    """Return whether a frame uses the canonical vocabulary."""

    return "timestamp_utc" in frame.columns and "water_level_m" in frame.columns


def is_model_schema(frame: pd.DataFrame) -> bool:
    """Return whether a frame uses the model (legacy Wai) vocabulary."""

    return "timestamp" in frame.columns and "water_level" in frame.columns


def to_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a canonical observation frame into the model schema.

    Canonical water levels are always meters, so the output carries
    ``units="m"``.  Extra canonical provenance columns (``record_type``,
    ``qc_status``, ``qc_flags``, ``retrieved_at``, ``latency_seconds``) are
    dropped because the model path does not consume them; numeric covariate
    columns beyond the canonical set are preserved so external forcing data
    survives the conversion.
    """

    if is_model_schema(frame):
        return frame.copy()
    if not is_canonical(frame):
        raise ValueError(
            "Frame matches neither the canonical nor the model schema; "
            f"columns: {list(frame.columns)}"
        )

    drop = {"record_type", "qc_status", "qc_flags", "retrieved_at", "latency_seconds"}
    out = frame.drop(columns=[c for c in drop if c in frame.columns]).rename(
        columns=CANONICAL_TO_MODEL
    )
    out["units"] = "m"
    return out.sort_values("timestamp").reset_index(drop=True)


def to_canonical_frame(
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    record_type: str = "observation",
) -> pd.DataFrame:
    """Convert a model-schema frame into the canonical vocabulary.

    ``source`` defaults to the frame's own ``source`` column when present.
    """

    if is_canonical(frame):
        missing = [c for c in CANONICAL_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Canonical frame missing columns: {missing}")
        return frame.copy()
    if not is_model_schema(frame):
        raise ValueError(
            "Frame matches neither the canonical nor the model schema; "
            f"columns: {list(frame.columns)}"
        )

    if source is None:
        if "source" in frame.columns and len(frame):
            source = str(frame["source"].iloc[0])
        else:
            source = "UNKNOWN"
    return canonicalize_frame(
        frame,
        source=source,
        record_type=record_type,
        qc_status_col="qc_status" if "qc_status" in frame.columns else None,
        qc_flags_col="qc_flags" if "qc_flags" in frame.columns else None,
    )
