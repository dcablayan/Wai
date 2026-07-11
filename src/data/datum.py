"""Vertical datum conversion for canonical observation frames.

Different gauge networks report on different vertical datums (MLLW, MSL,
NAVD88, STND, or vendor-local zero).  Wai previously failed closed on any mix
(:func:`src.data.canonicalize.assert_compatible_datums`), which blocks pairing
a NAVD88 gauge with an MLLW reference station outright.

Conversion between tidal datums at one station is a constant vertical offset
published per station (NOAA datums product, or a vendor survey).  This module
applies those per-station offsets when they are known and still fails closed
when they are not — an unverified conversion silently corrupts residuals.

Offsets are declared as ``{(station_id, from_datum, to_datum): offset_m}``
meaning ``level_to = level_from + offset_m``.  The reverse direction is
derived automatically.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from src.data.canonicalize import DatumMismatchError, assert_compatible_datums

DatumOffsets = Mapping[tuple[str, str, str], float]


def _normalize_key(station_id: str, from_datum: str, to_datum: str) -> tuple[str, str, str]:
    return (str(station_id), str(from_datum).upper(), str(to_datum).upper())


def lookup_offset_m(
    offsets: DatumOffsets,
    station_id: str,
    from_datum: str,
    to_datum: str,
) -> float | None:
    """Return the additive offset in meters, deriving the reverse if needed."""

    from_d = str(from_datum).upper()
    to_d = str(to_datum).upper()
    if from_d == to_d:
        return 0.0
    normalized = {_normalize_key(*k): float(v) for k, v in offsets.items()}
    direct = normalized.get(_normalize_key(station_id, from_d, to_d))
    if direct is not None:
        return direct
    reverse = normalized.get(_normalize_key(station_id, to_d, from_d))
    if reverse is not None:
        return -reverse
    return None


def convert_datum(
    frame: pd.DataFrame,
    *,
    to_datum: str,
    offsets: DatumOffsets,
) -> pd.DataFrame:
    """Return a copy of a canonical frame converted to ``to_datum``.

    Every (station, datum) group present in the frame must have a known
    offset; otherwise :class:`DatumMismatchError` is raised so an unverified
    conversion can never slip through.
    """

    if frame is None or frame.empty:
        return frame

    to_d = str(to_datum).upper()
    out = frame.copy()
    for (station, datum), group in out.groupby(["station_id", "datum"], sort=False):
        offset = lookup_offset_m(offsets, str(station), str(datum), to_d)
        if offset is None:
            raise DatumMismatchError(
                f"No datum offset configured for station {station!r}: "
                f"{str(datum).upper()} -> {to_d}"
            )
        if offset != 0.0:
            out.loc[group.index, "water_level_m"] = group["water_level_m"] + offset
    out["datum"] = to_d
    return out


def harmonize_datums(
    frames: Sequence[pd.DataFrame],
    *,
    to_datum: str | None = None,
    offsets: DatumOffsets | None = None,
    label: str = "canonical frames",
) -> list[pd.DataFrame]:
    """Convert frames onto one datum, or verify they already share one.

    With no ``offsets``, this is exactly the legacy fail-closed check.  With
    offsets it converts what it can and fails closed on anything unknown.
    Returns the (possibly converted) frames in input order; ``None``/empty
    entries pass through untouched.
    """

    live = [f for f in frames if f is not None and not f.empty]
    if not live:
        raise DatumMismatchError(f"{label}: no datum values available")

    if to_datum is None:
        datums = sorted(
            {str(v).upper() for f in live for v in f["datum"].dropna().unique()}
        )
        if len(datums) == 1:
            return list(frames)
        if not offsets:
            # Preserve the legacy error message and behavior.
            assert_compatible_datums(live, label=label)
        to_datum = datums[0]

    converted: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty:
            converted.append(frame)
            continue
        converted.append(convert_datum(frame, to_datum=to_datum, offsets=offsets or {}))
    assert_compatible_datums(
        [f for f in converted if f is not None and not f.empty], label=label
    )
    return converted
