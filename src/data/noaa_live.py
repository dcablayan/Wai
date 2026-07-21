"""Live NOAA CO-OPS snapshot assembly for the dashboard.

This module keeps provider access separate from Streamlit presentation.  It
never substitutes demo or mock records when a live request fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.canonicalize import assert_compatible_datums
from src.data.noaa import NOAACoopsAdapter
from src.data.noaa_catalog import NOAAStation, load_noaa_station_snapshot


SUPPORTED_LIVE_DATUMS = (
    "MLLW",
    "MSL",
    "MHHW",
    "NAVD",
    "STND",
    "IGLD",
    "LWD",
)


@dataclass(frozen=True)
class NOAALiveSnapshot:
    station: NOAAStation
    datum: str
    lookback_hours: int
    as_of: pd.Timestamp
    retrieved_at: pd.Timestamp
    frame: pd.DataFrame
    includes_tide_predictions: bool

    @property
    def latest_observed_at(self) -> pd.Timestamp:
        observed = self.frame.dropna(subset=["observed_m"])
        if observed.empty:
            raise ValueError("Live NOAA snapshot contains no observations")
        return pd.Timestamp(observed.iloc[-1]["timestamp"])

    @property
    def latest_observed_m(self) -> float:
        observed = self.frame.dropna(subset=["observed_m"])
        if observed.empty:
            raise ValueError("Live NOAA snapshot contains no observations")
        return float(observed.iloc[-1]["observed_m"])

    @property
    def latest_prediction_m(self) -> float:
        paired = self.frame.dropna(subset=["observed_m", "predicted_m"])
        if paired.empty:
            raise ValueError("Live NOAA snapshot contains no aligned prediction")
        return float(paired.iloc[-1]["predicted_m"])

    @property
    def latest_residual_m(self) -> float:
        paired = self.frame.dropna(subset=["residual_m"])
        if paired.empty:
            raise ValueError("Live NOAA snapshot contains no aligned residual")
        return float(paired.iloc[-1]["residual_m"])


@dataclass(frozen=True)
class NOAAOperationalGuidance:
    """Station-aligned NOAA OFS water-level guidance around the live timestamp."""

    station: NOAAStation
    datum: str
    history_hours: int
    forecast_hours: int
    as_of: pd.Timestamp
    retrieved_at: pd.Timestamp
    frame: pd.DataFrame


def fetch_live_noaa_snapshot(
    station_id: str,
    *,
    lookback_hours: int = 72,
    datum: str = "MLLW",
    include_tide_predictions: bool = True,
    as_of: object | None = None,
    adapter: NOAACoopsAdapter | None = None,
    station: NOAAStation | None = None,
) -> NOAALiveSnapshot:
    """Fetch and align live observations with NOAA tide predictions.

    The API is queried in metric units and GMT by :class:`NOAACoopsAdapter`.
    Exact six-minute timestamps are paired; missing provider points remain
    visible in the returned outer-joined frame instead of being interpolated.
    """

    station = _resolve_station(station_id, station)
    datum = str(datum).upper()
    if datum not in SUPPORTED_LIVE_DATUMS or datum not in station.datum_options:
        raise ValueError(
            f"Unsupported live datum {datum!r} for {station.label}; "
            f"expected one of {station.datum_options}"
        )
    if include_tide_predictions and not station.has_tide_predictions:
        raise ValueError(
            f"{station.label} does not publish NOAA astronomical tide predictions"
        )
    lookback_hours = int(lookback_hours)
    if not 1 <= lookback_hours <= 30 * 24:
        raise ValueError("lookback_hours must be between 1 and 720")

    finish = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if finish.tzinfo is None:
        finish = finish.tz_localize("UTC")
    else:
        finish = finish.tz_convert("UTC")
    finish = finish.floor("min")
    begin = finish - pd.Timedelta(hours=lookback_hours)

    client = adapter or NOAACoopsAdapter()
    observations = client.fetch_observations(
        station.station_id,
        begin,
        finish,
        latitude=station.latitude,
        longitude=station.longitude,
        datum=datum,
        use_cache=False,
    )
    observations = _clip_window(observations, begin, finish)
    if observations.empty:
        raise ValueError(
            f"NOAA returned no observations for {station.label} in the selected window"
        )
    observed = observations[
        ["timestamp_utc", "water_level_m", "qc_status"]
    ].rename(
        columns={
            "timestamp_utc": "timestamp",
            "water_level_m": "observed_m",
        }
    )
    if include_tide_predictions:
        predictions = client.fetch_tide_predictions(
            station.station_id,
            begin,
            finish,
            latitude=station.latitude,
            longitude=station.longitude,
            datum=datum,
            use_cache=False,
        )
        assert_compatible_datums(
            [observations, predictions], label="live NOAA observations and predictions"
        )
        predictions = _clip_window(predictions, begin, finish)
        if predictions.empty:
            raise ValueError(
                f"NOAA returned no tide predictions for {station.label} in the selected window"
            )
        predicted = predictions[["timestamp_utc", "water_level_m"]].rename(
            columns={
                "timestamp_utc": "timestamp",
                "water_level_m": "predicted_m",
            }
        )
        frame = pd.merge(
            observed, predicted, on="timestamp", how="outer"
        ).sort_values("timestamp")
    else:
        frame = observed.sort_values("timestamp").copy()
        frame["predicted_m"] = float("nan")
    frame["residual_m"] = frame["observed_m"] - frame["predicted_m"]
    frame = frame.reset_index(drop=True)
    if (
        include_tide_predictions
        and frame.dropna(subset=["observed_m", "predicted_m"]).empty
    ):
        raise ValueError(
            f"NOAA returned no timestamp-aligned observations and predictions for {station.label}"
        )

    retrieved = pd.to_datetime(observations["retrieved_at"], utc=True).max()
    return NOAALiveSnapshot(
        station=station,
        datum=datum,
        lookback_hours=lookback_hours,
        as_of=finish,
        retrieved_at=pd.Timestamp(retrieved),
        frame=frame,
        includes_tide_predictions=include_tide_predictions,
    )


def fetch_live_noaa_operational_guidance(
    station_id: str,
    *,
    history_hours: int = 24,
    forecast_hours: int = 48,
    datum: str = "MLLW",
    as_of: object | None = None,
    adapter: NOAACoopsAdapter | None = None,
    station: NOAAStation | None = None,
) -> NOAAOperationalGuidance:
    """Fetch NOAA OFS water-level guidance without a synthetic fallback.

    NOAA only offers ``ofs_water_level`` at stations within supported
    Operational Forecast System domains. Unsupported stations fail visibly so
    the dashboard never relabels astronomical tide predictions as OFS output.
    """

    station = _resolve_station(station_id, station)
    datum = str(datum).upper()
    if datum not in SUPPORTED_LIVE_DATUMS or datum not in station.datum_options:
        raise ValueError(
            f"Unsupported live datum {datum!r} for {station.label}; "
            f"expected one of {station.datum_options}"
        )
    history_hours = int(history_hours)
    forecast_hours = int(forecast_hours)
    if not 1 <= history_hours <= 30 * 24:
        raise ValueError("history_hours must be between 1 and 720")
    if not 1 <= forecast_hours <= 72:
        raise ValueError("forecast_hours must be between 1 and 72")

    current = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    current = current.floor("min")
    begin = current - pd.Timedelta(hours=history_hours)
    finish = current + pd.Timedelta(hours=forecast_hours)

    client = adapter or NOAACoopsAdapter()
    guidance = client.fetch_operational_forecast(
        station.station_id,
        begin,
        finish,
        latitude=station.latitude,
        longitude=station.longitude,
        datum=datum,
        use_cache=False,
    )
    guidance = _clip_window(guidance, begin, finish)
    if guidance.empty:
        raise ValueError(
            f"NOAA returned no OFS water-level guidance for {station.label}"
        )

    frame = guidance[["timestamp_utc", "water_level_m"]].rename(
        columns={
            "timestamp_utc": "timestamp",
            "water_level_m": "guidance_m",
        }
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    retrieved = pd.to_datetime(guidance["retrieved_at"], utc=True).max()
    return NOAAOperationalGuidance(
        station=station,
        datum=datum,
        history_hours=history_hours,
        forecast_hours=forecast_hours,
        as_of=current,
        retrieved_at=pd.Timestamp(retrieved),
        frame=frame,
    )


def _resolve_station(
    station_id: str,
    station: NOAAStation | None,
) -> NOAAStation:
    requested_id = str(station_id)
    if station is not None:
        if station.station_id != requested_id:
            raise ValueError(
                f"Station metadata ID {station.station_id} does not match {requested_id}"
            )
        return station
    resolved = load_noaa_station_snapshot().by_id.get(requested_id)
    if resolved is None:
        raise ValueError(f"Unsupported NOAA live station: {station_id}")
    return resolved


def _clip_window(
    frame: pd.DataFrame,
    begin: pd.Timestamp,
    finish: pd.Timestamp,
) -> pd.DataFrame:
    work = frame.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)
    return (
        work.loc[
            work["timestamp_utc"].between(begin, finish, inclusive="both")
        ]
        .drop_duplicates(subset=["timestamp_utc"], keep="last")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
