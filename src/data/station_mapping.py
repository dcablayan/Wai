"""Station pairing metadata for regional-to-local forecasts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationPair:
    """Map a local station to a regional NOAA station and residual transfer."""

    target_station_id: str
    paired_noaa_station_id: str
    target_name: str = ""
    noaa_name: str = ""
    residual_scale: float = 1.0
    lag_minutes: int = 0
    datum: str = "MLLW"


DEFAULT_STATION_PAIRS: dict[str, StationPair] = {
    "DEMO-HNL": StationPair(
        target_station_id="DEMO-HNL",
        paired_noaa_station_id="1612340",
        target_name="Demo Honolulu local station",
        noaa_name="Honolulu, HI",
        residual_scale=0.85,
        lag_minutes=0,
    ),
    "DEMO-SFO": StationPair(
        target_station_id="DEMO-SFO",
        paired_noaa_station_id="9414290",
        target_name="Demo San Francisco local station",
        noaa_name="San Francisco, CA",
        residual_scale=0.9,
        lag_minutes=0,
    ),
}


def get_station_pair(
    target_station_id: str,
    *,
    paired_noaa_station_id: str | None = None,
    mappings: dict[str, StationPair] | None = None,
) -> StationPair:
    """Return configured pairing, falling back to an explicit NOAA station."""

    pairs = mappings or DEFAULT_STATION_PAIRS
    if target_station_id in pairs:
        pair = pairs[target_station_id]
        if paired_noaa_station_id and paired_noaa_station_id != pair.paired_noaa_station_id:
            return StationPair(
                target_station_id=target_station_id,
                paired_noaa_station_id=paired_noaa_station_id,
                residual_scale=pair.residual_scale,
                lag_minutes=pair.lag_minutes,
                datum=pair.datum,
            )
        return pair
    if not paired_noaa_station_id:
        raise KeyError(
            f"No NOAA station pairing configured for {target_station_id!r}; "
            "provide paired_noaa_station_id explicitly."
        )
    return StationPair(
        target_station_id=target_station_id,
        paired_noaa_station_id=paired_noaa_station_id,
    )
