"""Refresh Wai's bundled nationwide NOAA active-station catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.noaa_catalog import (
    DEFAULT_NOAA_STATION_SNAPSHOT,
    fetch_noaa_station_catalog,
    save_noaa_station_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch every active NOAA CO-OPS water-level station and record which "
            "stations also publish astronomical tide predictions."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_NOAA_STATION_SNAPSHOT,
        help="JSON snapshot path (default: data/noaa_active_stations.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = fetch_noaa_station_catalog()
    target = save_noaa_station_catalog(catalog, args.output)
    print(
        f"Saved {catalog.count} active water-level stations to {target}\n"
        f"  Tide-prediction compatible: {catalog.tide_prediction_count}\n"
        f"  Great Lakes stations: {catalog.great_lakes_count}\n"
        f"  Regions: {len(catalog.regions)}\n"
        f"  Retrieved: {catalog.retrieved_at}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
