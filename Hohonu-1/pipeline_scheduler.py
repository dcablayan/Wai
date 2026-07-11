#!/usr/bin/env python3
"""Scheduled execution wrapper for the Hohonu prediction pipeline.

The scheduler repeatedly executes one or more nodes through
`hohonu_driver_script.run_pipeline` and writes each run as a JSON artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from hohonu_driver_script import run_pipeline
from tide_ml_engine import MODEL_FAMILY_HELP_TEXT


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _coerce_model_families(raw) -> Optional[List[str]]:
    if raw is None:
        return None

    tokens: List[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        for token in text.replace(" ", ",").split(","):
            token = token.strip().lower()
            if token:
                tokens.append(token)
    return tokens if tokens else None


@dataclass
class SchedulerConfig:
    node_names: List[str]
    strategy: str
    steps: int
    use_digital_twin: bool
    ensemble_size: int
    include_lstm: bool
    include_pinn: bool
    candidate_profile: str
    candidate_model_families: Optional[List[str]]
    candidate_mix_max_size: int
    meta_top_k: int
    meta_holdout_ratio: float


def _coerce_nodes(raw_nodes: List[str], nodes_file: Optional[Path]) -> List[str]:
    nodes = list(dict.fromkeys([n.strip() for n in raw_nodes if n.strip()]))
    if nodes_file is not None:
        if not nodes_file.exists():
            raise FileNotFoundError(f"nodes file not found: {nodes_file}")
        lines = []
        for line in nodes_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            lines.append(value)
        for value in lines:
            for token in value.replace(",", " ").split():
                token = token.strip()
                if token:
                    nodes.append(token)
        nodes = list(dict.fromkeys(nodes))
    if not nodes:
        raise ValueError("no node names provided (use positional nodes or --nodes-file)")
    return nodes


def _append_manifest_row(run_payload: dict, manifest_path: Path, node: str, status: str, result: Optional[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run_id",
                "run_utc",
                "node_name",
                "status",
                "strategy",
                "candidate_profile",
                "candidate_model_families",
                "model_name",
                "selected_rmse",
                "selected_mae",
                "selected_mape",
                "selected_r2",
                "selected_corr",
                "selected_mpe",
                "selected_me",
                "selected_minmax",
                "selected_nse",
                "selected_qa_score",
                "steps",
                "digital_twin_used",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_payload["run_id"],
                "run_utc": run_payload["run_utc"],
                "node_name": node,
                "status": status,
                "strategy": run_payload["strategy"],
                "candidate_profile": run_payload["candidate_profile"],
                "candidate_model_families": "|".join(
                    run_payload.get("candidate_model_families") or []
                ),
                "model_name": (result or {}).get("model_name"),
                "selected_rmse": (result or {}).get("selected_rmse"),
                "selected_mae": (result or {}).get("selected_mae"),
                "selected_mape": (result or {}).get("selected_mape"),
                "selected_r2": (result or {}).get("selected_r2"),
                "selected_corr": (result or {}).get("selected_corr"),
                "selected_mpe": (result or {}).get("selected_mpe"),
                "selected_me": (result or {}).get("selected_me"),
                "selected_minmax": (result or {}).get("selected_minmax"),
                "selected_nse": (result or {}).get("selected_nse"),
                "selected_qa_score": (result or {}).get("selected_qa_score"),
                "steps": run_payload["steps"],
                "digital_twin_used": (result or {}).get("digital_twin_used", False),
            }
        )


def _run_once(config: SchedulerConfig, out_dir: Path, run_counter: int) -> dict:
    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": run_id,
        "run_counter": run_counter,
        "run_utc": run_started.isoformat(),
        "strategy": config.strategy,
        "steps": config.steps,
        "candidate_profile": config.candidate_profile,
        "candidate_model_families": config.candidate_model_families,
        "candidate_mix_max_size": config.candidate_mix_max_size,
        "nodes": {},
    }

    for node_name in config.node_names:
        try:
            result = run_pipeline(
                node_name=node_name,
                model_strategy=config.strategy,
                steps=config.steps,
                use_digital_twin=config.use_digital_twin,
                return_metadata=True,
                ensemble_size=config.ensemble_size,
                include_lstm=config.include_lstm,
                include_pinn=config.include_pinn,
                candidate_profile=config.candidate_profile,
                candidate_model_families=config.candidate_model_families,
                candidate_mix_max_size=config.candidate_mix_max_size,
                meta_top_k=config.meta_top_k,
                meta_holdout_ratio=config.meta_holdout_ratio,
            )
            result = _to_jsonable(result)
            result["status"] = "ok"
            payload["nodes"][node_name] = result
            _append_manifest_row(payload, out_dir / "manifest.csv", node_name, "ok", result)
        except Exception as exc:
            payload["nodes"][node_name] = {"status": "error", "error": str(exc)}
            _append_manifest_row(payload, out_dir / "manifest.csv", node_name, "error", payload["nodes"][node_name])

    payload["run_complete_utc"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "ok" if all(
        isinstance(node_payload, dict) and node_payload.get("status") == "ok"
        for node_payload in payload["nodes"].values()
    ) else "partial_error"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{run_id}_{run_counter:04d}.json"
    out_path.write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    return {"path": str(out_path), "run_id": run_id, "payload": payload}


def _print_summary(output: dict, node_count: int) -> None:
    result = output["payload"]
    run_complete = result.get("run_complete_utc")
    print(f"[{run_complete}] Run #{output['run_id']} complete.")
    print(f"Nodes: {node_count}")
    for node, node_payload in result["nodes"].items():
        status = node_payload.get("status", "unknown")
        rmse = node_payload.get("selected_rmse")
        mae = node_payload.get("selected_mae")
        r2 = node_payload.get("selected_r2")
        nse = node_payload.get("selected_nse")
        qa_score = node_payload.get("selected_qa_score")
        model = node_payload.get("model_name", "unknown")
        if rmse is None:
            rmse = "n/a"
        metrics = []
        if mae is not None:
            metrics.append(f"mae={mae}")
        if r2 is not None:
            metrics.append(f"r2={r2}")
        minmax = node_payload.get("selected_minmax")
        if minmax is not None:
            metrics.append(f"minmax={minmax}")
        if nse is not None:
            metrics.append(f"nse={nse}")
        if qa_score is not None:
            metrics.append(f"qa={qa_score}")
        metric_str = ", ".join(metrics)
        if metric_str:
            print(f"  - {node}: {status} | {model} | rmse={rmse} | {metric_str}")
        else:
            print(f"  - {node}: {status} | {model} | rmse={rmse}")
    print(f"Saved -> {output['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hohonu pipeline on interval.")
    parser.add_argument("nodes", nargs="*", help="Node IDs to run.")
    parser.add_argument(
        "--nodes-file",
        type=Path,
        default=None,
        help="Optional file with node IDs (one per line).",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=3600,
        help="Seconds between scheduled runs.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one pass and exit.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum number of runs. 0 means unlimited.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./pipeline_runs"),
        help="Directory for JSON outputs and manifest.",
    )
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=["var", "auto", "auto-ml", "ensemble", "meta", "mix", "ml"],
        help="Prediction strategy.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=960,
        help="Forecast horizon.",
    )
    parser.add_argument(
        "--digital-twin",
        dest="use_digital_twin",
        action="store_true",
        default=True,
        help="Enable NOAA-based digital twin.",
    )
    parser.add_argument(
        "--no-digital-twin",
        dest="use_digital_twin",
        action="store_false",
        help="Disable NOAA-based digital twin.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Top-k ensemble size for strategy=ensemble.",
    )
    parser.add_argument(
        "--candidate-profile",
        type=str,
        default="compact",
        choices=["compact", "auto", "broad"],
        help="Candidate profile.",
    )
    parser.add_argument(
        "--model-families",
        type=str,
        nargs="*",
        default=None,
        help=MODEL_FAMILY_HELP_TEXT,
    )
    parser.add_argument(
        "--mix-size",
        type=int,
        default=4,
        help="Max number of models to include in mix strategy.",
    )
    parser.add_argument(
        "--meta-top-k",
        type=int,
        default=4,
        help="Top candidates used by meta-stacker.",
    )
    parser.add_argument(
        "--meta-holdout-ratio",
        type=float,
        default=0.20,
        help="Holdout ratio for meta-stacker fitting.",
    )
    parser.add_argument(
        "--include-lstm",
        action="store_true",
        help="Enable LSTM candidates.",
    )
    parser.add_argument(
        "--include-pinn",
        action="store_true",
        help="Enable PINN candidates.",
    )

    args = parser.parse_args()
    if args.nodes_file is None and not args.nodes:
        raise SystemExit("Provide one or more node IDs or --nodes-file.")

    node_names = _coerce_nodes(args.nodes, args.nodes_file)
    candidate_model_families = _coerce_model_families(args.model_families)

    config = SchedulerConfig(
        node_names=node_names,
        strategy=args.strategy,
        steps=args.steps,
        use_digital_twin=args.use_digital_twin,
        ensemble_size=args.ensemble_size,
        include_lstm=args.include_lstm,
        include_pinn=args.include_pinn,
        candidate_profile=args.candidate_profile,
        candidate_model_families=candidate_model_families,
        candidate_mix_max_size=max(2, args.mix_size),
        meta_top_k=args.meta_top_k,
        meta_holdout_ratio=args.meta_holdout_ratio,
    )

    run_count = 0
    max_runs = 1 if args.once else int(args.max_runs)
    if max_runs == 0:
        max_runs = 0
    out_dir = Path(args.out_dir).expanduser().resolve()

    print(f"Scheduling {len(node_names)} node(s). Strategy={config.strategy}.")
    print(f"Output directory: {out_dir}")

    while True:
        run_count += 1
        start = time.time()
        output = _run_once(config=config, out_dir=out_dir, run_counter=run_count)
        _print_summary(output, len(node_names))

        if max_runs and run_count >= max_runs:
            break

        sleep_interval = max(1, int(args.interval_seconds))
        elapsed = time.time() - start
        delay = max(0.0, sleep_interval - elapsed)
        if delay <= 0:
            continue
        print(f"Sleeping for {int(delay)}s before next run.")
        time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Scheduler stopped by user.")
