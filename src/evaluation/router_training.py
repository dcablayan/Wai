"""Train an advisory learned router from historical replay rows."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
        "training_metadata": metadata.to_dict(orient="records"),
        "audit": asdict(audit),
    }

    saved_model_path = None
    if model_path is not None:
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(artifact, handle)
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
    indices = np.arange(len(y))
    if len(y) < 8 or y.nunique() < 2:
        return indices, np.array([], dtype=int)

    label_counts = y.value_counts()
    stratify = y if label_counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=cfg.test_fraction,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


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
