"""Sanity checks for the multistep forecasting model.

Trains a small model on the bundled test-data fixtures and runs prediction
against each (historic, future) pair, asserting shape, finiteness,
non-negativity, and that the output index matches the future inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from simple_multistep_model import (
    DataFrameMultistepModel,
    ResidualBootstrapModel,
)

TEST_DATA = Path(__file__).parent.parent / "test-data"
INDEX_COLS = ["time_period", "location"]
TARGET = "disease_cases"
FEATURE_COLS = ["rainfall", "mean_temperature"]
N_TARGET_LAGS = 6
N_SAMPLES = 50

PREDICTION_DATES = [
    "2012-04-01",
    "2012-05-01",
    "2012-06-01",
    "2012-07-01",
    "2012-08-01",
    "2012-09-01",
    "2012-10-01",
]


def _train(training_csv: Path) -> DataFrameMultistepModel:
    data = pd.read_csv(training_csv)
    y = data[INDEX_COLS + [TARGET]]
    X = data[INDEX_COLS + FEATURE_COLS]
    one_step = ResidualBootstrapModel(
        RandomForestRegressor(
            n_estimators=20, max_depth=5, min_samples_leaf=5, random_state=42
        )
    )
    model = DataFrameMultistepModel(one_step, N_TARGET_LAGS, TARGET)
    model.fit(X, y)
    return model


def _predict(
    model: DataFrameMultistepModel, historic_csv: Path, future_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    historic = pd.read_csv(historic_csv)
    future = pd.read_csv(future_csv)
    n_steps = future.groupby("location").size().iloc[0]
    features = pd.concat(
        [historic[INDEX_COLS + FEATURE_COLS], future[INDEX_COLS + FEATURE_COLS]],
        ignore_index=True,
    ).sort_values(INDEX_COLS)
    y_historic = historic[INDEX_COLS + [TARGET]]
    preds = model.predict(y_historic, features, n_steps, N_SAMPLES)
    return preds, future


@pytest.fixture(scope="module")
def trained_model() -> DataFrameMultistepModel:
    return _train(TEST_DATA / "training_data.csv")


@pytest.fixture(scope="module")
def training_max_cases() -> float:
    data = pd.read_csv(TEST_DATA / "training_data.csv")
    return float(data[TARGET].max())


def test_training_data_present():
    """The bundled training data exists and has the expected schema."""
    data = pd.read_csv(TEST_DATA / "training_data.csv")
    for col in INDEX_COLS + FEATURE_COLS + [TARGET]:
        assert col in data.columns, f"missing column {col!r}"
    assert len(data) > 0


def test_training_runs(trained_model):
    """Training completes and exposes the configured lag count."""
    assert trained_model.n_target_lags == N_TARGET_LAGS


@pytest.mark.parametrize("date", PREDICTION_DATES)
def test_prediction_shape(trained_model, date):
    """Predictions have one row per (location, future timestep) and N_SAMPLES sample columns."""
    preds, future = _predict(
        trained_model,
        TEST_DATA / f"historic_data_{date}.csv",
        TEST_DATA / f"future_data_{date}.csv",
    )

    assert len(preds) == len(future)
    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    assert sample_cols == [f"sample_{i}" for i in range(N_SAMPLES)]
    assert set(preds.columns) == set(INDEX_COLS) | set(sample_cols)


@pytest.mark.parametrize("date", PREDICTION_DATES)
def test_prediction_index_matches_future(trained_model, date):
    """Output (time_period, location) pairs match the future-features input."""
    preds, future = _predict(
        trained_model,
        TEST_DATA / f"historic_data_{date}.csv",
        TEST_DATA / f"future_data_{date}.csv",
    )

    pred_keys = set(zip(preds["time_period"].astype(str), preds["location"].astype(str)))
    future_keys = set(zip(future["time_period"].astype(str), future["location"].astype(str)))
    assert pred_keys == future_keys


@pytest.mark.parametrize("date", PREDICTION_DATES)
def test_predictions_finite_and_nonnegative(trained_model, date):
    """Disease-case samples are finite, non-NaN, and non-negative (model clamps to >= 0)."""
    preds, _ = _predict(
        trained_model,
        TEST_DATA / f"historic_data_{date}.csv",
        TEST_DATA / f"future_data_{date}.csv",
    )

    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    samples = preds[sample_cols].to_numpy()
    assert not np.isnan(samples).any(), "predictions contain NaN"
    assert np.isfinite(samples).all(), "predictions contain inf"
    assert (samples >= 0).all(), "predictions contain negative disease counts"


@pytest.mark.parametrize("date", PREDICTION_DATES)
def test_predictions_within_plausible_range(trained_model, training_max_cases, date):
    """Predictions stay within an order-of-magnitude band around historical max.

    Catches the failure mode where recursive lag-feeding causes runaway
    explosion (predictions diverging far above anything seen in training).
    """
    preds, _ = _predict(
        trained_model,
        TEST_DATA / f"historic_data_{date}.csv",
        TEST_DATA / f"future_data_{date}.csv",
    )

    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    samples = preds[sample_cols].to_numpy()
    ceiling = max(training_max_cases * 10.0, 100.0)
    assert samples.max() <= ceiling, (
        f"max prediction {samples.max():.1f} exceeds plausibility ceiling {ceiling:.1f} "
        f"(training max was {training_max_cases:.1f})"
    )


@pytest.mark.parametrize("date", PREDICTION_DATES)
def test_predictions_have_variance(trained_model, date):
    """Across the N_SAMPLES draws, at least one row should show non-zero spread.

    A degenerate model that always returns the point estimate (no residual
    bootstrap) would produce identical samples — this guards against that.
    """
    preds, _ = _predict(
        trained_model,
        TEST_DATA / f"historic_data_{date}.csv",
        TEST_DATA / f"future_data_{date}.csv",
    )

    sample_cols = [c for c in preds.columns if c.startswith("sample_")]
    row_std = preds[sample_cols].to_numpy().std(axis=1)
    assert row_std.max() > 0, "all sample draws were identical across all rows"
