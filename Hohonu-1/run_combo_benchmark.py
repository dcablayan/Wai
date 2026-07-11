"""Convenience script to test multiple model-combination candidates for one node."""

from __future__ import annotations

import argparse
import json
import numpy as np
from typing import Dict, List

from hohonu_driver_script import _prepare_combined_series
from tide_ml_engine import (
    _HAS_TENSORFLOW,
    _HAS_TORCH,
    MODEL_FAMILY_HELP_TEXT,
    benchmark_best_model,
    benchmark_best_model_rolling,
    get_default_candidate_grid,
    run_auto_ml_search,
)


QA_DISPLAY_METRICS = [
    ("rmse_target", "rmse"),
    ("mae_target", "mae"),
    ("mape_target", "mape"),
    ("r2_target", "r2"),
    ("corr_target", "corr"),
    ("mpe_target", "mpe"),
    ("me_target", "me"),
    ("minmax_target", "minmax"),
    ("nse_target", "nse"),
    ("selected_qa_score", "qa_score"),
]

SELECTED_METRICS_TO_PRINT = (
    ("selected_rmse", "rmse"),
    ("selected_mae", "mae"),
    ("selected_mape", "mape"),
    ("selected_r2", "r2"),
    ("selected_corr", "corr"),
    ("selected_minmax", "minmax"),
    ("selected_nse", "nse"),
    ("selected_qa_score", "qa_score"),
)


def _coerce_model_families(raw):
    if raw is None:
        return None

    tokens = []
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        for token in text.replace(" ", ",").split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens if tokens else None


def _format_metrics(
    metrics: List[Dict], top_k: int = 5, sort_metric: str = "rmse_target"
) -> str:
    lines = []
    if not metrics:
        return "  (no benchmark rows)"
    sort_metric = (sort_metric or "rmse_target").strip()
    lines.append(f"Sorted by: {sort_metric}")

    higher_is_better = {
        "nse_target",
        "r2_target",
        "corr_target",
        "minmax_target",
        "selected_nse",
        "selected_r2",
        "selected_corr",
        "selected_minmax",
        "selected_qa_score",
        "qa_score",
    }

    def _metric_value(row: Dict):
        value = row.get("selected_qa_score") if sort_metric == "qa_score" else row.get(sort_metric)
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            return float("-inf") if sort_metric in higher_is_better else float("inf")
        return float(value)

    sorted_rows = sorted(
        metrics,
        key=_metric_value,
        reverse=(sort_metric in higher_is_better),
    )

    for idx, item in enumerate(sorted_rows[:top_k], start=1):
        metric_parts = []
        for key, label in QA_DISPLAY_METRICS:
            value = item.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                metric_parts.append(f"{label}={float(value):.4f}")
            else:
                metric_parts.append(f"{label}=n/a")
        lines.append(
            f"{idx:>2}. {item.get('name', 'unknown')}\n"
            f"    {' '.join(metric_parts)}"
        )
    return "\n".join(lines)


def _metric_to_str(value) -> str:
    if isinstance(value, (int, float, np.number)) and np.isfinite(value):
        return f"{float(value):.4f}"
    return "n/a"


def _format_family_health(
    rows: List[Dict], top_k: int = 6, sort_metric: str = "leaderboard_score"
) -> str:
    if not rows:
        return "  (no family health rows)"

    higher_is_better = {
        "avg_nse",
        "avg_r2",
        "avg_corr",
        "avg_minmax",
        "avg_qa_score",
        "leaderboard_score",
    }

    def _metric_value(row: Dict):
        value = row.get(sort_metric)
        if not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
            return float("-inf") if sort_metric in higher_is_better else float("inf")
        return float(value)

    rows_sorted = sorted(
        rows,
        key=_metric_value,
        reverse=(sort_metric in higher_is_better),
    )

    lines = [f"Sorted by: {sort_metric}"]
    for idx, row in enumerate(rows_sorted[:top_k], start=1):
        lines.append(
            f"{idx:>2}. {row.get('family', 'unknown')}"
            f" | folds={row.get('folds')} candidates={row.get('candidate_count')}"
            f" | avg_rmse={_metric_to_str(row.get('avg_rmse'))}"
            f" | avg_qa={_metric_to_str(row.get('avg_qa_score'))}"
            f" | leaderboard_score={_metric_to_str(row.get('leaderboard_score'))}"
        )
    return "\n".join(lines)


def _selected_summary(payload: Dict) -> Dict:
    if not payload:
        return {
            "selected_rmse": None,
            "selected_mae": None,
            "selected_mape": None,
            "selected_r2": None,
            "selected_corr": None,
            "selected_mpe": None,
            "selected_me": None,
            "selected_minmax": None,
            "selected_nse": None,
            "selected_qa_score": None,
        }

    return {
        "model_name": payload.get("model_name"),
        "selected_rmse": _metric_to_float(payload.get("selected_rmse")),
        "selected_mae": _metric_to_float(payload.get("selected_mae")),
        "selected_mape": _metric_to_float(payload.get("selected_mape")),
        "selected_r2": _metric_to_float(payload.get("selected_r2")),
        "selected_corr": _metric_to_float(payload.get("selected_corr")),
        "selected_mpe": _metric_to_float(payload.get("selected_mpe")),
        "selected_me": _metric_to_float(payload.get("selected_me")),
        "selected_minmax": _metric_to_float(payload.get("selected_minmax")),
        "selected_nse": _metric_to_float(payload.get("selected_nse")),
        "selected_qa_score": _metric_to_float(payload.get("selected_qa_score")),
        "strategy": payload.get("strategy"),
    }


def _selected_payload_to_string(payload: Dict) -> str:
    if not payload:
        return "n/a"
    parts = [
        f"{label}={payload.get(key):.6f}"
        for key, label in SELECTED_METRICS_TO_PRINT
        if isinstance(payload.get(key), (int, float, np.number))
        and np.isfinite(payload.get(key))
    ]
    return " | ".join(parts) if parts else "n/a"


def _metric_to_float(value):
    if isinstance(value, (int, float, np.number)):
        return float(value) if np.isfinite(value) else None
    return None


def _parse_qa_weights(raw_weights):
    if raw_weights is None:
        return None
    raw_weights = str(raw_weights).strip()
    if not raw_weights:
        return None
    parsed = {}
    for token in raw_weights.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "qa-weights must be formatted as key:value pairs, e.g. "
                "'rmse:0.4,mape:0.2,r2:0.15,nse:0.15'."
            )
        key, value = token.split(":", 1)
        key = key.strip().lower().replace("-", "_")
        parsed[key] = float(value.strip())
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate model combos for one Hohonu node."
    )
    parser.add_argument("node_name", help="Node id from Device table")
    parser.add_argument("--steps", type=int, default=960, help="Forecast horizon")
    parser.add_argument("--months", type=int, default=8, help="Lookback months")
    parser.add_argument("--offset", type=int, default=20, help="Lookback offset in days")
    parser.add_argument("--holdout", type=float, default=0.15, help="Holdout ratio for benchmark")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all", "best", "ensemble", "meta", "mix"],
        help="Which combo evaluation lanes to run",
    )
    parser.add_argument(
        "--include-lstm",
        dest="include_lstm",
        action="store_true",
        help="Include LSTM candidates if TensorFlow is available",
    )
    parser.add_argument(
        "--include-pinn",
        dest="include_pinn",
        action="store_true",
        help="Include PINN candidates if PyTorch is available",
    )
    parser.add_argument(
        "--candidate-profile",
        type=str,
        default="compact",
        choices=["compact", "auto", "broad"],
        help="Candidate search profile: compact, auto (subsampled broad), or broad",
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
        "--ensemble-sizes",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Ensemble sizes to evaluate",
    )
    parser.add_argument(
        "--meta-top-k",
        type=int,
        default=4,
        help="Number of top candidates to feed meta-stacker",
    )
    parser.add_argument(
        "--meta-holdout-ratio",
        type=float,
        default=0.20,
        help="Holdout ratio used to fit meta-stacker",
    )
    parser.add_argument(
        "--no-digital-twin",
        action="store_true",
        help="Disable NOAA digital-twin forcing for LSTM/PINN/auto search",
    )
    parser.add_argument(
        "--qa-sort-metric",
        type=str,
        default="rmse_target",
        choices=[
            "rmse_target",
            "mae_target",
            "mape_target",
            "r2_target",
            "corr_target",
            "mpe_target",
            "me_target",
            "minmax_target",
            "nse_target",
            "qa_score",
        ],
        help="Metric used to sort benchmark leaderboard rows.",
    )
    parser.add_argument(
        "--rolling-backtest",
        action="store_true",
        help="Run rolling holdout backtest and return family health report.",
    )
    parser.add_argument(
        "--rolling-folds",
        type=int,
        default=4,
        help="How many rolling folds to evaluate.",
    )
    parser.add_argument(
        "--rolling-window",
        type=str,
        default="auto",
        choices=["auto", "month", "quarter"],
        help="Rolling backtest holdout window size.",
    )
    parser.add_argument(
        "--holdout-steps",
        type=int,
        default=None,
        help="Fixed holdout step count for rolling backtest windows.",
    )
    parser.add_argument(
        "--family-health-sort-metric",
        type=str,
        default="leaderboard_score",
        choices=[
            "leaderboard_score",
            "avg_qa_score",
            "avg_rmse",
            "avg_mae",
            "avg_nse",
            "std_rmse",
            "rank_stability",
        ],
        help="Metric used to sort family-health rows.",
    )
    parser.add_argument(
        "--qa-weights",
        type=str,
        default=None,
        help=(
            "Optional QA metric weights for composite score, as key:value pairs "
            "(for example: 'rmse:0.45,mape:0.2,r2:0.2,nse:0.15'). "
            "Defaults use built-in weights."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON payload only",
    )

    args = parser.parse_args()

    include_lstm = args.include_lstm and _HAS_TENSORFLOW
    include_pinn = args.include_pinn and _HAS_TORCH
    if args.include_lstm and not _HAS_TENSORFLOW:
        print("Requested LSTM but TensorFlow is not available; skipping LSTM candidates.")
    if args.include_pinn and not _HAS_TORCH:
        print("Requested PINN but PyTorch is not available; skipping PINN candidates.")
    candidate_model_families = _coerce_model_families(args.model_families)

    qa_metric_weights = _parse_qa_weights(args.qa_weights)

    candidate_grid = get_default_candidate_grid(
        include_lstm=include_lstm,
        include_pinn=include_pinn,
        profile=args.candidate_profile,
    )

    combined_df, noaa_name = _prepare_combined_series(
        args.node_name, number_of_months=args.months, offset_days_backward=args.offset
    )

    benchmark = benchmark_best_model(
        combined_df,
        candidate_grid=candidate_grid,
        holdout_ratio=args.holdout,
        random_state=0,
        candidate_model_families=candidate_model_families,
        qa_metric_weights=qa_metric_weights,
    )

    rolling_backtest = None
    if args.rolling_backtest:
        try:
            rolling_backtest = benchmark_best_model_rolling(
                combined_df,
                candidate_grid=candidate_grid,
                holdout_ratio=args.holdout,
                rolling_folds=args.rolling_folds,
                random_state=0,
                candidate_model_families=candidate_model_families,
                candidate_profile=args.candidate_profile,
                qa_metric_weights=qa_metric_weights,
                backtest_window=args.rolling_window,
                holdout_steps=args.holdout_steps,
            )
        except Exception as exc:
            rolling_backtest = {
                "error": str(exc),
                "backtest_window": args.rolling_window,
                "rows": [],
                "family_health": [],
            }

    single = None
    if args.mode in {"all", "best"}:
        single = run_auto_ml_search(
            combined_df,
            steps=args.steps,
            candidate_grid=candidate_grid,
            target_col=-1,
            random_state=0,
            use_digital_twin=not args.no_digital_twin,
            strategy="best",
            candidate_model_families=candidate_model_families,
            qa_metric_weights=qa_metric_weights,
        )

    meta = None
    if args.mode in {"all", "meta"}:
        meta = run_auto_ml_search(
            combined_df,
            steps=args.steps,
            candidate_grid=candidate_grid,
            target_col=-1,
            random_state=0,
            use_digital_twin=not args.no_digital_twin,
            strategy="meta",
            meta_top_k=args.meta_top_k,
            meta_holdout_ratio=args.meta_holdout_ratio,
            candidate_model_families=candidate_model_families,
            qa_metric_weights=qa_metric_weights,
        )

    ensemble_runs = {}
    if args.mode in {"all", "ensemble"}:
        for size in args.ensemble_sizes:
            size = int(size)
            if size < 2:
                continue
            ensemble_runs[size] = run_auto_ml_search(
                combined_df,
                steps=args.steps,
                candidate_grid=candidate_grid,
                target_col=-1,
                random_state=0,
                use_digital_twin=not args.no_digital_twin,
                strategy="ensemble",
                ensemble_size=size,
                candidate_model_families=candidate_model_families,
                qa_metric_weights=qa_metric_weights,
            )

    mix = None
    if args.mode in {"all", "mix"}:
        mix = run_auto_ml_search(
            combined_df,
            steps=args.steps,
            candidate_grid=candidate_grid,
            target_col=-1,
            random_state=0,
            use_digital_twin=not args.no_digital_twin,
            strategy="mix",
            candidate_mix_max_size=args.mix_size,
            candidate_model_families=candidate_model_families,
            qa_metric_weights=qa_metric_weights,
        )

    result = {
        "node_name": args.node_name,
        "noaa_name": noaa_name,
        "flags": {
            "tensorflow_available": _HAS_TENSORFLOW,
            "torch_available": _HAS_TORCH,
            "requested_lstm": args.include_lstm,
            "requested_pinn": args.include_pinn,
            "used_lstm": include_lstm,
            "used_pinn": include_pinn,
            "candidate_profile": args.candidate_profile,
            "candidate_model_families": candidate_model_families,
            "digital_twin": not args.no_digital_twin,
            "qa_sort_metric": args.qa_sort_metric,
            "rolling_backtest": args.rolling_backtest,
            "rolling_folds": args.rolling_folds if args.rolling_backtest else None,
            "rolling_window": args.rolling_window if args.rolling_backtest else None,
            "holdout_steps": args.holdout_steps,
            "family_health_sort_metric": args.family_health_sort_metric,
            "qa_metric_weights": qa_metric_weights,
        },
        "benchmark": benchmark,
        "rolling_backtest": rolling_backtest,
        "single": _selected_summary(single),
        "meta": _selected_summary(meta),
        "ensemble": {
            str(k): _selected_summary(value)
            for k, value in ensemble_runs.items()
        },
        "mix": _selected_summary(mix),
        "scoreboards": {
            "single": single["scores"] if single else [],
            "meta": meta["scores"] if meta else [],
            "ensemble": {str(k): (v["scores"] if isinstance(v, dict) else []) for k, v in ensemble_runs.items()},
            "mix": mix["scores"] if mix else [],
        },
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"=== Combo Benchmark for {args.node_name} ===")
    print(f"NOAA: {noaa_name}")
    print("Dependency flags:")
    print(f"  TensorFlow available: {_HAS_TENSORFLOW}, requested: {args.include_lstm}")
    print(f"  PyTorch  available: {_HAS_TORCH},  requested: {args.include_pinn}")
    print(f"  Candidate profile: {args.candidate_profile}")
    if candidate_model_families is not None:
        print(f"  Candidate families: {candidate_model_families}")
    print(f"  Candidate count: {len(candidate_grid)}")
    print("\nTop holdout performers:")
    print(_format_metrics(benchmark, top_k=8, sort_metric=args.qa_sort_metric))
    if rolling_backtest is not None:
        print("\nRolling-backtest summary:")
        if rolling_backtest.get("error"):
            print(f"  error: {rolling_backtest.get('error')}")
        else:
            print(
                "  folds=generated/requested/evaluated: "
                f"{rolling_backtest.get('fold_count')}/"
                f"{rolling_backtest.get('requested_folds')}/"
                f"{rolling_backtest.get('evaluated_folds')} "
                f"window={rolling_backtest.get('backtest_window')} "
                f"holdout_steps={rolling_backtest.get('holdout_steps')}"
            )
            print(_format_metrics(rolling_backtest.get("rows") or [], top_k=8, sort_metric=args.qa_sort_metric))
            print(
                _format_family_health(
                    rolling_backtest.get("family_health") or [],
                    sort_metric=args.family_health_sort_metric,
                )
            )
    if single:
        print("\nSingle-best forecast config:")
        print(f"  {single.get('model_name')} | {_selected_payload_to_string(single)}")
    if meta:
        print("\nMeta-stacked forecast config:")
        print(f"  {meta.get('model_name')} | {_selected_payload_to_string(meta)}")
    for size, payload in ensemble_runs.items():
        print(
            f"Ensemble-{size}: {payload.get('model_name')} | "
            + _selected_payload_to_string(payload)
        )
    if mix:
        print("\nModel-mix forecast config:")
        print(f"  {mix.get('model_name')} | {_selected_payload_to_string(mix)}")


if __name__ == "__main__":
    main()
