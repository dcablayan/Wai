"""Train an advisory learned router from historical replay rows."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.artifacts import encode_router_artifact, save_json_artifact
from src.orchestration.learned_router import _feature_row


REQUIRED_REPLAY_COLUMNS = {
    "forecast_origin_utc",
    "target_time_utc",
    "context_features",
    "missing_data_conditions",
    "actual_m",
    "error_by_expert",
    "max_hohonu_input_time_utc",
    "max_noaa_input_time_utc",
}

FORBIDDEN_FEATURE_TOKENS = (
    "actual",
    "error",
    "target_time",
    "forecast_error",
    "expert_prediction",
)


class ReplayAuditError(ValueError):
    """Raised when a replay dataset is not safe for router training."""


@dataclass(frozen=True)
class RouterTrainingConfig:
    """Training configuration for the first supervised router."""

    max_depth: int = 4
    min_samples_leaf: int = 1
    test_fraction: float = 0.25
    random_state: int = 42
    min_training_rows: int = 4


@dataclass
class ReplayAuditReport:
    """Leakage and schema audit result."""

    n_rows: int
    n_violations: int
    violations: list[str]
    required_columns_present: bool
    feature_columns_checked: list[str]


@dataclass
class RouterTrainingReport:
    """Summary of a trained advisory router."""

    n_rows: int
    n_train: int
    n_test: int
    label_counts: dict[str, int]
    train_accuracy: float
    test_accuracy: float | None
    feature_columns: list[str]
    audit: dict[str, Any]
    model_path: str | None = None


def audit_replay_for_router_training(
    replay: pd.DataFrame,
    *,
    raise_on_violation: bool = True,
) -> ReplayAuditReport:
    """Check replay rows for leakage before supervised training."""

    violations: list[str] = []
    missing_columns = sorted(REQUIRED_REPLAY_COLUMNS - set(replay.columns))
    if missing_columns:
        violations.append(f"missing required columns: {missing_columns}")

    feature_columns_checked: list[str] = []
    if not missing_columns:
        for idx, row in replay.iterrows():
            origin = _timestamp(row["forecast_origin_utc"])
            target = _timestamp(row["target_time_utc"])
            if target <= origin:
                violations.append(f"row {idx}: target_time_utc is not after forecast_origin_utc")

            for col in ("max_hohonu_input_time_utc", "max_noaa_input_time_utc"):
                if pd.isna(row[col]):
                    continue
                max_input = _timestamp(row[col])
                if max_input > origin:
                    violations.append(f"row {idx}: {col} is after forecast origin")

            context_features = _parse_json(row["context_features"], label=f"row {idx} context_features")
            missing_conditions = _parse_json(row["missing_data_conditions"], label=f"row {idx} missing_data_conditions")
            feature_row = _feature_row(context_features, missing_conditions)
            feature_columns_checked.extend(feature_row.keys())
            forbidden = [
                key for key in feature_row
                if any(token in key.lower() for token in FORBIDDEN_FEATURE_TOKENS)
            ]
            if forbidden:
                violations.append(f"row {idx}: forbidden leakage-like feature names: {forbidden}")

    report = ReplayAuditReport(
        n_rows=int(len(replay)),
        n_violations=len(violations),
        violations=violations,
        required_columns_present=not missing_columns,
        feature_columns_checked=sorted(set(feature_columns_checked)),
    )
    if violations and raise_on_violation:
        raise ReplayAuditError("; ".join(violations))
    return report


def build_router_training_frame(
    replay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return encoded features, best-expert labels, and row metadata."""

    audit_replay_for_router_training(replay)
    feature_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    for idx, row in replay.iterrows():
        if pd.isna(row["actual_m"]):
            continue
        errors = _parse_json(row["error_by_expert"], label=f"row {idx} error_by_expert")
        label = best_expert_label(errors)
        if label is None:
            continue

        context_features = _parse_json(row["context_features"], label=f"row {idx} context_features")
        missing_conditions = _parse_json(row["missing_data_conditions"], label=f"row {idx} missing_data_conditions")
        feature_rows.append(_feature_row(context_features, missing_conditions))
        labels.append(label)
        metadata_rows.append({
            "forecast_origin_utc": row["forecast_origin_utc"],
            "target_time_utc": row["target_time_utc"],
            "best_expert": label,
            "best_abs_error_m": abs(float(errors[label])),
        })

    if not feature_rows:
        raise ReplayAuditError("No replay rows had actuals and finite expert errors")

    raw_features = pd.DataFrame(feature_rows)
    encoded = pd.get_dummies(raw_features, dummy_na=True)
    y = pd.Series(labels, name="best_expert")
    metadata = pd.DataFrame(metadata_rows)
    return encoded, y, metadata


def train_router_from_replay(
    replay: pd.DataFrame,
    *,
    config: RouterTrainingConfig | None = None,
    model_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> tuple[dict[str, Any], RouterTrainingReport]:
    """Train and optionally persist an advisory learned-router artifact."""

    cfg = config or RouterTrainingConfig()
    audit = audit_replay_for_router_training(replay)
    X, y, metadata = build_router_training_frame(replay)
    if len(X) < cfg.min_training_rows:
        raise ReplayAuditError(
            f"Need at least {cfg.min_training_rows} training rows, got {len(X)}"
        )

    train_idx, test_idx = _train_test_indices(y, cfg)
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx] if len(test_idx) else X.iloc[[]]
    y_test = y.iloc[test_idx] if len(test_idx) else y.iloc[[]]

    model = DecisionTreeClassifier(
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_accuracy = None
    if len(X_test):
        test_accuracy = float(accuracy_score(y_test, model.predict(X_test)))

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "classes": list(model.classes_),
        "config": asdict(cfg),
        "n_train": int(len(X_train)),
        "validation": "forward_time",
        "training_metadata": metadata.to_dict(orient="records"),
        "audit": asdict(audit),
    }

    saved_model_path = None
    if model_path is not None:
        path = Path(model_path)
        save_json_artifact(
            path,
            encode_router_artifact(artifact),
            kind="learned_router",
        )
        saved_model_path = str(path)

    report = RouterTrainingReport(
        n_rows=int(len(X)),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        label_counts={str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        train_accuracy=float(accuracy_score(y_train, train_pred)),
        test_accuracy=test_accuracy,
        feature_columns=list(X.columns),
        audit=asdict(audit),
        model_path=saved_model_path,
    )

    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))

    return artifact, report


@dataclass
class RouterPolicyEvaluation:
    """Forward-validated comparison of routing policies on replay rows."""

    n_test: int
    validation: str
    oracle_mae: float | None
    rule_router_mae: float | None
    learned_router_mae: float | None
    routing_regret_m: float | None
    learned_avg_expert_calls: float
    rule_avg_expert_calls: float | None
    fallback_rate: float | None
    unavailable_rate: float | None


def evaluate_router_policies(
    replay: pd.DataFrame,
    *,
    config: RouterTrainingConfig | None = None,
) -> RouterPolicyEvaluation:
    """Compare oracle, rule, and learned routing with no future leakage.

    A per-expert absolute-error regressor is trained on origin-time features +
    expert identity (training fold only).  On the held-out fold the learned
    router picks ``argmin`` predicted error; we report its realised MAE against
    the oracle (best achievable per origin) and the rule/cascade's realised MAE.
    """

    cfg = config or RouterTrainingConfig()
    rows = []
    for _, row in replay.iterrows():
        if pd.isna(row.get("actual_m")):
            continue
        errors = _parse_json(row["error_by_expert"], label="error_by_expert")
        abs_errors = {
            k: abs(float(v)) for k, v in errors.items()
            if _is_finite(v)
        }
        if not abs_errors:
            continue
        rows.append({
            "origin": _timestamp(row["forecast_origin_utc"]),
            "station": row.get("target_station_id", "*"),
            "features": _feature_row(
                _parse_json(row["context_features"], label="context_features"),
                _parse_json(row["missing_data_conditions"], label="missing_data_conditions"),
            ),
            "abs_errors": abs_errors,
            "selected": _safe_list(row.get("selected_experts")),
            "forecast_error_m": row.get("forecast_error_m"),
            "expert_calls": row.get("expert_calls"),
            "fallback_used": bool(row.get("fallback_used", False)),
            "status": row.get("result_status", "available"),
        })

    if len(rows) < 4:
        return RouterPolicyEvaluation(
            n_test=0, validation="insufficient_rows", oracle_mae=None,
            rule_router_mae=None, learned_router_mae=None, routing_regret_m=None,
            learned_avg_expert_calls=1.0, rule_avg_expert_calls=None,
            fallback_rate=None, unavailable_rate=None,
        )

    stations = {r["station"] for r in rows}
    if len(stations) > 1:
        validation = "station_held_out"
        held = sorted(stations)[-1]
        train_rows = [r for r in rows if r["station"] != held]
        test_rows = [r for r in rows if r["station"] == held]
    else:
        validation = "forward_time"
        rows.sort(key=lambda r: r["origin"])
        n_test = max(1, int(round(len(rows) * cfg.test_fraction)))
        train_rows = rows[: len(rows) - n_test]
        test_rows = rows[len(rows) - n_test:]

    if not train_rows or not test_rows:
        return RouterPolicyEvaluation(
            n_test=0, validation=validation, oracle_mae=None,
            rule_router_mae=None, learned_router_mae=None, routing_regret_m=None,
            learned_avg_expert_calls=1.0, rule_avg_expert_calls=None,
            fallback_rate=None, unavailable_rate=None,
        )

    # Long-form per-(origin, expert) training frame.
    train_records = []
    train_targets = []
    for r in train_rows:
        for expert, err in r["abs_errors"].items():
            rec = dict(r["features"])
            rec["expert"] = expert
            train_records.append(rec)
            train_targets.append(err)
    X_train = pd.get_dummies(pd.DataFrame(train_records), dummy_na=True)
    feature_columns = list(X_train.columns)
    model = DecisionTreeRegressor(max_depth=cfg.max_depth, random_state=cfg.random_state)
    model.fit(X_train, np.array(train_targets, dtype=float))

    learned_errs, oracle_errs, rule_errs = [], [], []
    rule_calls = []
    for r in test_rows:
        candidates = list(r["abs_errors"].keys())
        recs = []
        for expert in candidates:
            rec = dict(r["features"])
            rec["expert"] = expert
            recs.append(rec)
        Xc = pd.get_dummies(pd.DataFrame(recs), dummy_na=True).reindex(
            columns=feature_columns, fill_value=0
        )
        predicted = model.predict(Xc)
        learned_pick = candidates[int(np.argmin(predicted))]
        learned_errs.append(r["abs_errors"][learned_pick])
        oracle_errs.append(min(r["abs_errors"].values()))
        if r["forecast_error_m"] is not None and _is_finite(r["forecast_error_m"]):
            rule_errs.append(abs(float(r["forecast_error_m"])))
        if r["expert_calls"] is not None and _is_finite(r["expert_calls"]):
            rule_calls.append(float(r["expert_calls"]))

    oracle_mae = float(np.mean(oracle_errs)) if oracle_errs else None
    learned_mae = float(np.mean(learned_errs)) if learned_errs else None
    rule_mae = float(np.mean(rule_errs)) if rule_errs else None
    regret = (learned_mae - oracle_mae) if (learned_mae is not None and oracle_mae is not None) else None
    return RouterPolicyEvaluation(
        n_test=len(test_rows),
        validation=validation,
        oracle_mae=oracle_mae,
        rule_router_mae=rule_mae,
        learned_router_mae=learned_mae,
        routing_regret_m=regret,
        learned_avg_expert_calls=1.0,
        rule_avg_expert_calls=float(np.mean(rule_calls)) if rule_calls else None,
        fallback_rate=float(np.mean([r["fallback_used"] for r in test_rows])),
        unavailable_rate=float(np.mean([r["status"] != "available" for r in test_rows])),
    )


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:  # noqa: BLE001
        return []


def best_expert_label(errors: dict[str, Any]) -> str | None:
    """Return the expert with the smallest absolute finite error."""

    candidates = []
    for expert, error in errors.items():
        try:
            value = float(error)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            candidates.append((abs(value), str(expert)))
    if not candidates:
        return None
    return sorted(candidates)[0][1]


def load_replay_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _train_test_indices(y: pd.Series, cfg: RouterTrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Forward-time split: the first rows train, the most recent rows test.

    Replay rows are produced in increasing forecast-origin order, so a positional
    split is a forward-in-time (rolling-origin) holdout — never a random split,
    which would leak future information into the training fold of a time series.
    """

    n = len(y)
    indices = np.arange(n)
    if n < 8 or y.nunique() < 2:
        return indices, np.array([], dtype=int)
    n_test = max(1, int(round(n * cfg.test_fraction)))
    n_train = n - n_test
    if n_train < 1:
        return indices, np.array([], dtype=int)
    return indices[:n_train], indices[n_train:]


def _parse_json(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ReplayAuditError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ReplayAuditError(f"{label} must decode to an object")
    return parsed


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
