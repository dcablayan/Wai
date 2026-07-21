"""Advisory learned router loaded from historical replay training artifacts.

The learned router is intentionally advisory.  The rule-based capability gate
and safety checks remain authoritative: a learned model may *rank* safe,
eligible experts, but it never overrides missing data, failed QC, datum
mismatch, or hard physical restrictions.  In production the learned router runs
in **shadow mode** — it records what it would have selected without controlling
the live route — and falls back to the rule router when:

- the model artifact is missing,
- the feature schema differs,
- inputs look out of distribution,
- sample support is too small,
- predicted utilities are too close to call, or
- the model raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.artifacts import decode_router_artifact, load_json_artifact


@dataclass(frozen=True)
class LearnedRouterPrediction:
    """A learned-router recommendation for one forecast context."""

    recommended_expert: str
    probabilities: dict[str, float]
    feature_values: dict[str, Any]


@dataclass(frozen=True)
class ShadowRecommendation:
    """What the learned router would select, recorded without controlling route."""

    recommended_expert: str | None
    source: str
    probabilities: dict[str, float] = field(default_factory=dict)


class LearnedRouter:
    """Load a supervised router artifact and predict the best-expert label."""

    def __init__(self, artifact: dict[str, Any], *, min_margin: float = 0.1) -> None:
        self.artifact = artifact
        self.model = artifact["model"]
        self.feature_columns = list(artifact["feature_columns"])
        self.classes = list(getattr(self.model, "classes_", artifact.get("classes", [])))
        self.min_margin = min_margin
        self.min_training_rows = int(artifact.get("config", {}).get("min_training_rows", 0) or 0)
        self.n_train = int(artifact.get("n_train", 0) or 0)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRouter":
        payload = load_json_artifact(path, expected_kind="learned_router")
        return cls(decode_router_artifact(payload))

    def predict_from_features(
        self,
        context_features: dict[str, Any],
        missing_data_conditions: dict[str, Any] | None = None,
    ) -> LearnedRouterPrediction:
        row = _feature_row(context_features, missing_data_conditions or {})
        encoded = pd.get_dummies(pd.DataFrame([row]), dummy_na=True)
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0)
        expert = str(self.model.predict(encoded)[0])
        probabilities: dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(encoded)[0]
            probabilities = {
                str(label): float(prob)
                for label, prob in zip(self.model.classes_, probs)
            }
        return LearnedRouterPrediction(
            recommended_expert=expert,
            probabilities=probabilities,
            feature_values=row,
        )

    def shadow_recommend(self, context) -> ShadowRecommendation:
        """Advisory recommendation for a live context (records only).

        Returns the recommended expert plus a ``source`` describing whether the
        learned model was trusted or a fallback condition triggered.
        """

        features = router_features_from_context(context)
        missing = router_missing_from_context(context)
        try:
            pred = self.predict_from_features(features, missing)
        except Exception:  # noqa: BLE001
            return ShadowRecommendation(None, "fallback_model_error", {})

        if self.n_train and self.n_train < max(self.min_training_rows, 8):
            return ShadowRecommendation(pred.recommended_expert, "fallback_low_support", pred.probabilities)

        probs = pred.probabilities
        if probs:
            ordered = sorted(probs.values(), reverse=True)
            margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
            if margin < self.min_margin:
                return ShadowRecommendation(pred.recommended_expert, "fallback_low_margin", probs)
        return ShadowRecommendation(pred.recommended_expert, "learned", probs)


def router_features_from_context(context) -> dict[str, Any]:
    """Origin-time features only (no actuals, errors, or expert predictions)."""

    return {
        "horizon_minutes": context.horizon_minutes,
        "hohonu_freshness_seconds": context.observation_freshness_seconds.get("hohonu"),
        "noaa_freshness_seconds": context.observation_freshness_seconds.get("noaa"),
        "hohonu_qc_status": context.qc_status.get("hohonu"),
        "noaa_qc_status": context.qc_status.get("noaa"),
        "recent_hohonu_trend_m_per_hour": context.recent_hohonu_trend_m_per_hour,
        "recent_noaa_residual_m": context.recent_noaa_residual_m,
        "noaa_residual_trend_m_per_hour": context.noaa_residual_trend_m_per_hour,
        "tide_phase": context.tide_phase,
    }


def router_missing_from_context(context) -> dict[str, Any]:
    return {
        "missing_latest_hohonu": context.latest_hohonu_observation is None,
        "missing_latest_noaa": context.latest_noaa_observation is None,
        "missing_tide_prediction": context.noaa_tide_prediction is None,
        "hohonu_qc_ok": context.hohonu_qc_ok,
        "noaa_qc_ok": context.noaa_qc_ok,
    }


def _feature_row(
    context_features: dict[str, Any],
    missing_data_conditions: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in sorted(context_features.items()):
        row[f"context__{key}"] = value
    for key, value in sorted(missing_data_conditions.items()):
        row[f"missing__{key}"] = value
    return row
