"""Tests for RunConfig YAML loading and end-to-end train/predict via the CLI helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from simple_multistep_model import RunConfig, load_run_config

# train/predict live at the repo root and are not a package - import via path.
import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import train as train_module  # noqa: E402
import predict as predict_module  # noqa: E402

TEST_DATA = REPO_ROOT / "test-data"
PERIOD = "2012-04-01"


def test_empty_mapping_yields_defaults(tmp_path: Path):
    """An empty `{}` mapping is a valid RunConfig (all fields have defaults)."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("{}\n")
    assert load_run_config(empty) == RunConfig()


def test_yaml_overrides_apply(tmp_path: Path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "feature_columns": ["rainfall"],
                "n_target_lags": 4,
                "n_samples": 7,
                "prob_wrapper": "bucketedresidual",
                "rf": {"n_estimators": 11, "max_depth": 3, "random_state": 1},
            }
        )
    )

    cfg = load_run_config(yaml_path)
    assert cfg.feature_columns == ["rainfall"]
    assert cfg.n_target_lags == 4
    assert cfg.n_samples == 7
    assert cfg.prob_wrapper == "bucketedresidual"
    assert cfg.rf.n_estimators == 11
    assert cfg.rf.max_depth == 3
    assert cfg.rf.random_state == 1
    # Untouched fields keep defaults
    assert cfg.target_variable == "disease_cases"
    assert cfg.min_bucket_size == RunConfig().min_bucket_size


def test_unknown_field_rejected(tmp_path: Path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(yaml.safe_dump({"not_a_real_field": 1}))
    with pytest.raises(Exception):
        load_run_config(yaml_path)


@pytest.mark.parametrize(
    "prob_wrapper", ["bucketedresidual", "bootstrap", "cross-conformal"]
)
def test_train_predict_end_to_end_with_yaml(tmp_path: Path, prob_wrapper: str):
    """train.train and predict.predict run against the test fixtures with a YAML config."""
    cfg_path = tmp_path / "run_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "feature_columns": ["rainfall", "mean_temperature"],
                "n_target_lags": 6,
                "n_samples": 25,
                "prob_wrapper": prob_wrapper,
                "min_bucket_size": 3,
                "rf": {
                    "n_estimators": 20,
                    "max_depth": 5,
                    "min_samples_leaf": 5,
                    "random_state": 0,
                },
            }
        )
    )

    model_path = tmp_path / "model.pkl"
    train_module.train(
        str(TEST_DATA / "training_data.csv"),
        str(model_path),
        str(cfg_path),
    )
    assert model_path.exists() and model_path.stat().st_size > 0

    out_path = tmp_path / "predictions.csv"
    predict_module.predict(
        str(model_path),
        str(TEST_DATA / f"historic_data_{PERIOD}.csv"),
        str(TEST_DATA / f"future_data_{PERIOD}.csv"),
        str(out_path),
        str(cfg_path),
    )

    preds = pd.read_csv(out_path)
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    assert sample_cols == [f"sample_{i}" for i in range(25)]

    future = pd.read_csv(TEST_DATA / f"future_data_{PERIOD}.csv")
    assert len(preds) == len(future)

    samples = preds[sample_cols].to_numpy()
    assert np.isfinite(samples).all()
    assert (samples >= 0).all()


def test_train_with_no_config_path_uses_defaults_for_features(tmp_path: Path):
    """Without a config, train falls back to the default feature_columns.

    The bundled test data lacks `mean_relative_humidity`, so the default
    config should fail with a clear KeyError - this guards against silently
    accepting incompatible data.
    """
    model_path = tmp_path / "model.pkl"
    with pytest.raises(KeyError):
        train_module.train(str(TEST_DATA / "training_data.csv"), str(model_path), None)
