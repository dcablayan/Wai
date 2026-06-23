"""Run one orchestrated forecast from mocked Hohonu and NOAA inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.data.station_mapping import StationPair
from src.forecasting import ForecastPipeline
from src.orchestration.context import build_forecast_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Wai's forecast orchestrator.")
    parser.add_argument("--station-id", default="HOHONU_TEST")
    parser.add_argument("--noaa-station-id", default="NOAA_TEST")
    parser.add_argument("--forecast-time", default="2024-01-01T18:00:00Z")
    parser.add_argument("--horizon-minutes", type=int, default=360)
    parser.add_argument("--mode", choices=["mini", "ultra", "legacy"], default="mini")
    args = parser.parse_args(argv)

    hohonu = mock_hohonu_observations(args.station_id, periods=300)
    noaa_obs = mock_noaa_observations(args.noaa_station_id, periods=300, residual_m=0.12)
    noaa_tide = mock_noaa_tide_predictions(args.noaa_station_id, periods=420)

    context = build_forecast_context(
        target_station_id=args.station_id,
        paired_noaa_station_id=args.noaa_station_id,
        horizon_minutes=args.horizon_minutes,
        forecast_time_utc=args.forecast_time,
        hohonu_observations=hohonu,
        noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide,
        station_pair=StationPair(args.station_id, args.noaa_station_id),
    )
    result = ForecastPipeline(mode=args.mode).run(context)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
