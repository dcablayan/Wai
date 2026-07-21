"""Regression tests for the supported legacy Hohonu execution surfaces."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


LEGACY_DIR = Path(__file__).resolve().parent.parent / "Hohonu-1"


@pytest.fixture(scope="module")
def legacy_path():
    sys.path.insert(0, str(LEGACY_DIR))
    try:
        yield
    finally:
        sys.path.remove(str(LEGACY_DIR))


def test_node_name_validation_blocks_path_traversal(legacy_path):
    driver = importlib.import_module("hohonu_driver_script")
    assert driver.validate_node_name("honolulu.pier-1") == "honolulu.pier-1"
    for unsafe in ("../secret", "station/../../secret", "/absolute", "bad name"):
        with pytest.raises(ValueError, match="node_name"):
            driver.validate_node_name(unsafe)


def test_var_window_selection_uses_forward_holdout_then_refits(legacy_path, monkeypatch):
    module = importlib.import_module("VAR_prediction")

    class FakeFit:
        def __init__(self, frame):
            self.values = frame.to_numpy(dtype=float)

        def forecast(self, y, steps):
            increments = np.arange(1, steps + 1, dtype=float)[:, None]
            return self.values[-1][None, :] + increments

    class FakeVAR:
        def __init__(self, frame):
            self.frame = frame

        def fit(self, lag):
            assert lag == 248
            return FakeFit(self.frame)

    monkeypatch.setattr(module, "VAR", FakeVAR)
    values = np.arange(7203, dtype=float)
    data = pd.DataFrame({"local": values, "noaa": values + 100.0})
    forecast = module.predict_water_level(data, steps=3)
    assert forecast[:, 0].tolist() == [7203.0, 7204.0, 7205.0]
    assert forecast[:, 1].tolist() == [7303.0, 7304.0, 7305.0]


def test_var_requires_a_real_forward_holdout(legacy_path):
    module = importlib.import_module("VAR_prediction")
    data = pd.DataFrame(np.ones((7200, 2)))
    assert module.predict_water_level(data, steps=3) is None
