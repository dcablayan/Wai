"""Forecast from any tide gauge's data — CSV export, catalog station, or both.

This is the provider-agnostic entry point. Point it at a delimited export
from any gauge (mapping column names on the command line), or at a station
registered in a catalog file, and it produces an orchestrated forecast:

    # A bare CSV with vendor column names, no NOAA counterpart:
    python -m scripts.run_gauge_forecast \
        --csv my_gauge.csv --timestamp-col time --water-level-col level_ft \
        --units ft --datum MLLW --station-id MY-GAUGE-01 \
        --horizon-minutes 180

    # Same gauge plus a reference station export and its tide predictions:
    python -m scripts.run_gauge_forecast \
        --csv my_gauge.csv --station-id MY-GAUGE-01 \
        --reference-csv noaa_export.csv --reference-station-id 1612340 \
        --tide-csv noaa_predictions.csv

    # A station described in a catalog file:
    python -m scripts.run_gauge_forecast --catalog data/stations.json \
        --station-id MY-GAUGE-01

Observations are regularized first (cadence inference, MAD despiking, small-
gap interpolation), so irregular sampling and outages are handled before the
orchestrator sees the data. With no tide product supplied, the harmonic
fallback expert fits constituents to the gauge's own history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datum import harmonize_datums
from src.data.regularize import regularize_frame
from src.data.schema import to_canonical_frame
from src.data.sources import CSVSource, get_source
from src.data.station_catalog import StationCatalog
from src.data.station_mapping import StationPair
from src.forecasting import ForecastPipeline
from src.orchestration.context import build_forecast_context


def _column_map(args: argparse.Namespace) -> dict[str, str]:
    mapping = {}
    if args.timestamp_col:
        mapping["timestamp"] = args.timestamp_col
    if args.water_level_col:
        mapping["water_level"] = args.water_level_col
    if args.station_id_col:
        mapping["station_id"] = args.station_id_col
    return mapping


def _load_csv(path: str, args: argparse.Namespace, *, record_type: str,
              station_id: str, source_label: str) -> pd.DataFrame:
    source = CSVSource(
        path,
        source_label=source_label,
        record_type=record_type,
        column_map=_column_map(args),
        units=args.units,
        datum=args.datum,
        latitude=args.lat,
        longitude=args.lon,
        station_id=station_id,
    )
    return source.fetch_observations(station_id, None, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an orchestrated forecast from any gauge's data."
    )
    parser.add_argument("--station-id", required=True, help="Target gauge station id")
    parser.add_argument("--csv", help="Local gauge observations (any delimited export)")
    parser.add_argument("--catalog", help="Station catalog JSON (alternative to --csv)")
    parser.add_argument("--reference-csv", help="Regional reference observations CSV")
    parser.add_argument("--reference-station-id", help="Regional reference station id")
    parser.add_argument("--tide-csv", help="Tide predictions CSV for the reference station")
    parser.add_argument("--timestamp-col", help="Name of the timestamp column in the CSVs")
    parser.add_argument("--water-level-col", help="Name of the water-level column")
    parser.add_argument("--station-id-col", help="Name of the station-id column, if any")
    parser.add_argument("--units", default="m", help="Water-level units in the file (m or ft)")
    parser.add_argument("--datum", default="MLLW", help="Vertical datum of the gauge")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--forecast-time", default=None,
                        help="Forecast origin (default: last observation)")
    parser.add_argument("--horizon-minutes", type=int, default=180)
    parser.add_argument("--mode", choices=["mini", "ultra", "legacy"], default="mini")
    parser.add_argument("--cadence-minutes", type=float, default=None,
                        help="Native sampling interval (default: inferred)")
    parser.add_argument("--no-despike", action="store_true")
    args = parser.parse_args(argv)

    if not args.csv and not args.catalog:
        parser.error("Provide --csv or --catalog")

    reference_id = args.reference_station_id
    if args.catalog:
        catalog = StationCatalog.load(args.catalog)
        meta = catalog.get(args.station_id)
        source = get_source(meta.source, **dict(meta.source_config))
        local = source.fetch_observations(args.station_id, None, None)
        reference_id = reference_id or meta.reference_station_id
        station_pair = meta.to_station_pair()
        cadence = args.cadence_minutes or meta.cadence_minutes
        offsets = meta.datum_offset_table()
    else:
        local = _load_csv(
            args.csv, args,
            record_type="observation",
            station_id=args.station_id,
            source_label="LOCAL_GAUGE",
        )
        station_pair = StationPair(
            target_station_id=args.station_id,
            paired_noaa_station_id=reference_id or args.station_id,
            datum=args.datum.upper(),
        )
        cadence = args.cadence_minutes
        offsets = {}

    local, report = regularize_frame(
        local, cadence_minutes=cadence, despike=not args.no_despike
    )
    local = local.dropna(subset=["water_level_m"]).reset_index(drop=True)

    reference = None
    if args.reference_csv:
        if not reference_id:
            parser.error("--reference-csv requires --reference-station-id")
        reference = _load_csv(
            args.reference_csv, args,
            record_type="observation",
            station_id=reference_id,
            source_label="REFERENCE",
        )
    tide = None
    if args.tide_csv:
        tide = _load_csv(
            args.tide_csv, args,
            record_type="tide_prediction",
            station_id=reference_id or args.station_id,
            source_label="TIDE_PREDICTIONS",
        )
        tide = to_canonical_frame(tide)

    frames = harmonize_datums(
        [local, reference, tide],
        to_datum=station_pair.datum if offsets else None,
        offsets=offsets,
        label="gauge forecast inputs",
    )
    local, reference, tide = frames

    forecast_time = args.forecast_time or local["timestamp_utc"].iloc[-1]

    context = build_forecast_context(
        target_station_id=args.station_id,
        paired_noaa_station_id=reference_id,
        horizon_minutes=args.horizon_minutes,
        forecast_time_utc=forecast_time,
        local_observations=local,
        regional_observations=reference,
        regional_tide_predictions=tide,
        station_pair=station_pair,
    )
    result = ForecastPipeline(mode=args.mode).run(context)
    payload = result.to_dict()
    payload["input_regularization"] = {
        "cadence_minutes": report.cadence_minutes,
        "records_in": report.n_input,
        "records_on_grid": report.n_output,
        "spikes_removed": report.n_spikes_removed,
        "interpolated": report.n_interpolated,
        "gap_rows": report.n_gap_rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
