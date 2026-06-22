"""Train an advisory learned router from a historical replay CSV."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.router_training import (
    RouterTrainingConfig,
    load_replay_csv,
    train_router_from_replay,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train Wai's advisory learned router from historical replay rows."
    )
    parser.add_argument("--replay", default="reports/routing_replay_mock.csv")
    parser.add_argument("--model-output", default="reports/router_model.pkl")
    parser.add_argument("--report-output", default="reports/router_training_report.json")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--min-training-rows", type=int, default=4)
    args = parser.parse_args(argv)

    replay = load_replay_csv(args.replay)
    _, report = train_router_from_replay(
        replay,
        config=RouterTrainingConfig(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            min_training_rows=args.min_training_rows,
        ),
        model_path=args.model_output,
        report_path=args.report_output,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
