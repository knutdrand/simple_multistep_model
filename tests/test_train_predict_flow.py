"""End-to-end checks for the train.py / predict.py CLI flow.

Drives the public `train()` and `predict()` entry points (the same ones the
argparse CLI calls) with the bundled `test-data/debug_config.yaml`, so any
regression in the YAML-config plumbing or the train/predict scripts shows
up here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import predict as predict_module  # noqa: E402
import train as train_module  # noqa: E402

from simple_multistep_model import load_run_config  # noqa: E402

TEST_DATA = REPO_ROOT / "test-data"
DEBUG_CONFIG = TEST_DATA / "debug_config.yaml"

PREDICTION_DATES = [
    "2012-04-01",
    "2012-05-01",
    "2012-06-01",
    "2012-07-01",
    "2012-08-01",
    "2012-09-01",
    "2012-10-01",
]


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train once per module against the debug config and reuse the pickle."""
    out_dir = tmp_path_factory.mktemp("train")
    model_path = out_dir / "model.pkl"
    train_module.train(
        str(TEST_DATA / "training_data.csv"),
        str(model_path),
        str(DEBUG_CONFIG),
    )
    return model_path


def test_debug_config_loads_cleanly():
    """The checked-in debug config validates against RunConfig."""
    cfg = load_run_config(DEBUG_CONFIG)
    assert cfg.feature_columns == ["rainfall", "mean_temperature"]
    assert cfg.use_residual_bucketing is True
    assert cfg.rf.n_estimators == 5


def test_train_creates_pickle(trained_model_path: Path):
    assert trained_model_path.exists()
    assert trained_model_path.stat().st_size > 0


@pytest.mark.parametrize("date", PREDICTION_DATES)
def test_predict_writes_expected_csv(
    trained_model_path: Path, tmp_path: Path, date: str
):
    """predict.predict produces a CSV whose schema and row count match the future input."""
    out_path = tmp_path / f"predictions_{date}.csv"
    predict_module.predict(
        str(trained_model_path),
        str(TEST_DATA / f"historic_data_{date}.csv"),
        str(TEST_DATA / f"future_data_{date}.csv"),
        str(out_path),
        str(DEBUG_CONFIG),
    )

    assert out_path.exists()
    preds = pd.read_csv(out_path)

    cfg = load_run_config(DEBUG_CONFIG)
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    assert sample_cols == [f"sample_{i}" for i in range(cfg.n_samples)]
    assert set(preds.columns) == {"time_period", "location"} | set(sample_cols)

    future = pd.read_csv(TEST_DATA / f"future_data_{date}.csv")
    assert len(preds) == len(future)

    pred_keys = set(zip(preds["time_period"].astype(str), preds["location"].astype(str)))
    future_keys = set(zip(future["time_period"].astype(str), future["location"].astype(str)))
    assert pred_keys == future_keys


@pytest.mark.parametrize("date", PREDICTION_DATES)
def test_predict_values_are_finite_and_nonnegative(
    trained_model_path: Path, tmp_path: Path, date: str
):
    """Bucketed path clamps to >= 0; debug config selects it, so this must hold."""
    out_path = tmp_path / f"predictions_{date}.csv"
    predict_module.predict(
        str(trained_model_path),
        str(TEST_DATA / f"historic_data_{date}.csv"),
        str(TEST_DATA / f"future_data_{date}.csv"),
        str(out_path),
        str(DEBUG_CONFIG),
    )

    preds = pd.read_csv(out_path)
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    samples = preds[sample_cols].to_numpy()
    assert np.isfinite(samples).all()
    assert (samples >= 0).all()


