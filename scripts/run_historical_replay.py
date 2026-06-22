"""Generate an offline historical replay dataset from mocked inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hohonu import mock_hohonu_observations
from src.data.noaa import mock_noaa_observations, mock_noaa_tide_predictions
from src.evaluation import HistoricalReplayConfig, run_historical_replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Wai historical replay with offline fixtures.")
    parser.add_argument("--station-id", default="HOHONU_TEST")
    parser.add_argument("--noaa-station-id", default="NOAA_TEST")
    parser.add_argument("--output", default="reports/routing_replay_mock.csv")
    parser.add_argument("--horizon-minutes", type=int, default=360)
    parser.add_argument("--step-minutes", type=int, default=180)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    hohonu = mock_hohonu_observations(args.station_id, periods=720)
    noaa_obs = mock_noaa_observations(args.noaa_station_id, periods=720, residual_m=0.12)
    noaa_tide = mock_noaa_tide_predictions(args.noaa_station_id, periods=900)

    replay = run_historical_replay(
        target_station_id=args.station_id,
        paired_noaa_station_id=args.noaa_station_id,
        hohonu_observations=hohonu,
        noaa_observations=noaa_obs,
        noaa_tide_predictions=noaa_tide,
        config=HistoricalReplayConfig(
            horizon_minutes=args.horizon_minutes,
            step_minutes=args.step_minutes,
        ),
    )
    replay.to_csv(output, index=False)
    print(f"Saved {len(replay)} replay rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
