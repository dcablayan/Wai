"""Prepared, indexed station data for fast leakage-safe context construction.

The first orchestrator rebuilt a :class:`ForecastContext` from raw DataFrames at
every forecast origin: it copied each frame, re-parsed ``timestamp_utc``,
re-applied a boolean mask, re-sorted, and computed the recent NOAA residual
trend with an ``iterrows`` loop that called a nearest-record search per row
(``O(n*m)``).  Profiling showed that residual-trend loop alone was ~72% of
context-build time.

``PreparedStationData`` does the normalize/sort/validate/align work **once** per
station, stores numpy timestamp arrays for ``searchsorted`` slicing, and
precomputes the NOAA observed-minus-tide residual series with a single vectorized
``merge_asof``.  Each forecast origin then becomes a handful of ``O(log n)``
lookups plus a small recent-window slice — no full-frame copies, no per-origin
``to_datetime``/``sort``, and no Python-level residual loop.

Leakage safety is preserved: every per-origin slice uses ``timestamp_utc <=
forecast_time`` (strictly no future observations enter features).  Tide
predictions are deterministic schedules and may extend past the origin, exactly
as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.canonicalize import assert_compatible_datums
from src.data.station_mapping import StationPair, get_station_pair

_OBS_COLUMNS = ("timestamp_utc", "water_level_m", "qc_status", "source", "datum")


def _prepare_frame(
    frame: pd.DataFrame | None,
    *,
    station_id: str,
    record_type: str,
) -> pd.DataFrame:
    """Filter to one station/record_type, normalize timestamps, sort once."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_OBS_COLUMNS))
    df = frame
    # Only copy the rows we keep, and only convert timestamps once.
    mask = (df["station_id"].astype(str) == str(station_id)) & (
        df["record_type"] == record_type
    )
    subset = df.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=list(_OBS_COLUMNS))
    subset["timestamp_utc"] = pd.to_datetime(subset["timestamp_utc"], utc=True)
    subset = subset.sort_values("timestamp_utc").reset_index(drop=True)
    return subset


def _ts_array(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty(0, dtype="datetime64[ns]")
    return frame["timestamp_utc"].to_numpy(dtype="datetime64[ns]")


def _nearest_index(ts: np.ndarray, when: pd.Timestamp) -> int | None:
    """Index of the record whose timestamp is nearest ``when`` (abs distance)."""

    if ts.size == 0:
        return None
    target = np.datetime64(when.tz_convert("UTC").tz_localize(None), "ns")
    pos = int(np.searchsorted(ts, target, side="left"))
    if pos == 0:
        return 0
    if pos >= ts.size:
        return ts.size - 1
    before = target - ts[pos - 1]
    after = ts[pos] - target
    return pos if after < before else pos - 1


def _upper_bound(ts: np.ndarray, when: pd.Timestamp) -> int:
    """Number of records with ``timestamp_utc <= when`` (leakage cutoff)."""

    if ts.size == 0:
        return 0
    target = np.datetime64(when.tz_convert("UTC").tz_localize(None), "ns")
    return int(np.searchsorted(ts, target, side="right"))


@dataclass
class PreparedStationData:
    """Indexed canonical frames for one target/NOAA station pair."""

    target_station_id: str
    paired_noaa_station_id: str
    station_pair: StationPair
    datum: str
    hohonu: pd.DataFrame
    noaa: pd.DataFrame
    tide: pd.DataFrame
    local_tide: pd.DataFrame
    hohonu_ts: np.ndarray
    noaa_ts: np.ndarray
    tide_ts: np.ndarray
    local_tide_ts: np.ndarray
    noaa_residual: pd.DataFrame  # timestamp_utc, residual_m (aligned obs - tide)
    noaa_residual_ts: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        target_station_id: str,
        paired_noaa_station_id: str | None,
        hohonu_observations: pd.DataFrame,
        noaa_observations: pd.DataFrame,
        noaa_tide_predictions: pd.DataFrame,
        station_pair: StationPair | None = None,
        local_tide_predictions: pd.DataFrame | None = None,
    ) -> "PreparedStationData":
        pair = station_pair or get_station_pair(
            target_station_id, paired_noaa_station_id=paired_noaa_station_id
        )
        hohonu = _prepare_frame(
            hohonu_observations, station_id=target_station_id, record_type="observation"
        )
        noaa = _prepare_frame(
            noaa_observations,
            station_id=pair.paired_noaa_station_id,
            record_type="observation",
        )
        tide = _prepare_frame(
            noaa_tide_predictions,
            station_id=pair.paired_noaa_station_id,
            record_type="tide_prediction",
        )
        local_tide = _prepare_frame(
            local_tide_predictions,
            station_id=target_station_id,
            record_type="tide_prediction",
        )
        datum = assert_compatible_datums(
            [f for f in (hohonu, noaa, tide, local_tide) if not f.empty],
            label="forecast context",
        )
        residual = _align_residual(noaa, tide)
        return cls(
            target_station_id=target_station_id,
            paired_noaa_station_id=pair.paired_noaa_station_id,
            station_pair=pair,
            datum=datum,
            hohonu=hohonu,
            noaa=noaa,
            tide=tide,
            local_tide=local_tide,
            hohonu_ts=_ts_array(hohonu),
            noaa_ts=_ts_array(noaa),
            tide_ts=_ts_array(tide),
            local_tide_ts=_ts_array(local_tide),
            noaa_residual=residual,
            noaa_residual_ts=_ts_array(residual),
        )

    # -- per-origin lookups -------------------------------------------------

    def latest_before(self, frame: pd.DataFrame, ts: np.ndarray, when: pd.Timestamp):
        n = _upper_bound(ts, when)
        if n == 0:
            return None
        return frame.iloc[n - 1].to_dict()

    def recent_slice(
        self, frame: pd.DataFrame, ts: np.ndarray, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        hi = _upper_bound(ts, end)
        lo = _upper_bound(ts, start - pd.Timedelta(nanoseconds=1))
        if hi <= lo:
            return frame.iloc[0:0]
        return frame.iloc[lo:hi]

    def nearest(self, frame: pd.DataFrame, ts: np.ndarray, when: pd.Timestamp):
        idx = _nearest_index(ts, when)
        if idx is None:
            return None
        return frame.iloc[idx].to_dict()

    def residual_at(self, when: pd.Timestamp) -> float | None:
        """Nearest observed NOAA residual at or before ``when`` (leakage-safe).

        Used by the regional-to-local expert for genuine lag application: it
        looks up the residual that was actually observed at the lagged source
        time instead of reusing the current residual.
        """

        ts = self.noaa_residual_ts
        if ts.size == 0:
            return None
        n = _upper_bound(ts, when)
        if n == 0:
            return None
        return float(self.noaa_residual.iloc[n - 1]["residual_m"])


def _align_residual(noaa: pd.DataFrame, tide: pd.DataFrame) -> pd.DataFrame:
    """Vectorized NOAA observed-minus-nearest-tide residual series."""

    if noaa.empty or tide.empty:
        return pd.DataFrame(columns=["timestamp_utc", "residual_m"])
    left = noaa[["timestamp_utc", "water_level_m"]].rename(
        columns={"water_level_m": "obs_m"}
    )
    right = tide[["timestamp_utc", "water_level_m"]].rename(
        columns={"water_level_m": "tide_m"}
    )
    merged = pd.merge_asof(
        left, right, on="timestamp_utc", direction="nearest"
    )
    merged["residual_m"] = merged["obs_m"] - merged["tide_m"]
    out = merged.dropna(subset=["residual_m"])[["timestamp_utc", "residual_m"]]
    return out.reset_index(drop=True)
