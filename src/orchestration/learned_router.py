"""Advisory learned router loaded from historical replay training artifacts."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LearnedRouterPrediction:
    """A learned-router recommendation for one forecast context."""

    recommended_expert: str
    probabilities: dict[str, float]
    feature_values: dict[str, Any]


class LearnedRouter:
    """Load a supervised router artifact and predict the best expert label.

    This class is intentionally advisory. The rule-based router remains the
    production default until a replay dataset from reviewed real station pairs
    proves that learned routing improves skill and safety.
    """

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.model = artifact["model"]
        self.feature_columns = list(artifact["feature_columns"])
        self.classes = list(getattr(self.model, "classes_", artifact.get("classes", [])))

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRouter":
        with open(path, "rb") as handle:
            artifact = pickle.load(handle)
        return cls(artifact)

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
