"""Example: composing pieces into a working multistep forecasting CLI.

Shows how to wire together:
1. A plain data-transform function (DataFrame -> DataFrame)
2. An sklearn regressor
3. A bootstrap wrapper (ResidualBootstrapModel)
4. The MultistepModel for recursive forecasting
5. A CLI for train/predict from CSV

Run:
    python -m simple_multistep_model.example train-cmd --help
    python -m simple_multistep_model.example train-cmd train.csv model.pkl
    python -m simple_multistep_model.example predict-cmd model.pkl historic.csv future.csv predictions.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import GradientBoostingRegressor

from simple_multistep_model.cli import create_cli_app
from simple_multistep_model.multistep import (
    MultistepModel,
    target_to_xarray,
    features_to_xarray,
    future_features_to_xarray,
)
from simple_multistep_model.one_step_model import ResidualBootstrapModel

# ---------------------------------------------------------------------------
# Configuration (just plain variables — no Pydantic config needed)
# ---------------------------------------------------------------------------

N_TARGET_LAGS = 12
N_SAMPLES = 200
TARGET_VARIABLE = "disease_cases"


# ---------------------------------------------------------------------------
# 1. Data transform: plain function, DataFrame -> DataFrame
# ---------------------------------------------------------------------------


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to the raw data. Customize this for your use case."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["time_period"]).dt.month
    return df


# ---------------------------------------------------------------------------
# 2. Train function
# ---------------------------------------------------------------------------


def train(data: pd.DataFrame) -> dict:
    """Train a multistep model from a training DataFrame.

    Args:
        data: Long-format DataFrame with [time_period, location, disease_cases, ...].

    Returns:
        Pickleable dict with trained model.
    """
    data = transform_data(data)

    index_cols = ["time_period", "location"]
    feature_cols = [c for c in data.columns if c not in index_cols + [TARGET_VARIABLE]]

    y_df = data[index_cols + [TARGET_VARIABLE]]
    X_df = data[index_cols + feature_cols] if feature_cols else data[index_cols]

    y_xr = target_to_xarray(y_df, TARGET_VARIABLE)
    X_xr = features_to_xarray(X_df)

    sklearn_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    one_step = ResidualBootstrapModel(sklearn_model)
    model = MultistepModel(one_step, N_TARGET_LAGS)
    model.fit_multi(y_xr, X_xr)

    return {"model": model, "feature_cols": feature_cols}


# ---------------------------------------------------------------------------
# 3. Predict function
# ---------------------------------------------------------------------------


def predict(model_dict: dict, historic: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions from trained model.

    Args:
        model_dict: Trained model dict from train().
        historic: Historic data in long format.
        future: Future periods in long format.

    Returns:
        DataFrame with [time_period, location, samples] columns.
    """
    model: MultistepModel = model_dict["model"]
    feature_cols: list[str] = model_dict["feature_cols"]

    historic = transform_data(historic)
    future = transform_data(future)

    index_cols = ["time_period", "location"]
    y_xr = target_to_xarray(historic[index_cols + [TARGET_VARIABLE]], TARGET_VARIABLE)
    previous_y = y_xr.isel(time=slice(-model.n_target_lags, None))

    X_future_xr = future_features_to_xarray(future[index_cols + feature_cols]) if feature_cols else None

    n_steps = future.groupby("location").size().iloc[0]
    predictions = model.predict_multi(previous_y, n_steps, N_SAMPLES, X_future_xr)

    return _xarray_predictions_to_pandas(predictions, future)


def _xarray_predictions_to_pandas(
    predictions: xr.DataArray, future_df: pd.DataFrame
) -> pd.DataFrame:
    """Convert xarray predictions to pandas long format with samples column."""
    future_df = future_df.copy()
    original_time_strs = future_df["time_period"].astype(str)
    future_df["_original_time"] = original_time_strs
    future_df["time_period"] = pd.to_datetime(future_df["time_period"])

    results_time: list[str] = []
    results_location: list[str] = []
    results_samples: list[list[float]] = []

    locations = predictions.coords["location"].values
    for loc in locations:
        loc_str = str(loc)
        loc_subset = future_df[future_df["location"] == loc_str].sort_values(by="time_period")
        loc_original_times = loc_subset["_original_time"].values

        loc_preds = predictions.sel(location=loc)
        n_steps = loc_preds.sizes["step"]

        for step_idx in range(n_steps):
            samples = loc_preds.isel(step=step_idx).values.tolist()
            results_time.append(str(loc_original_times[step_idx]))
            results_location.append(loc_str)
            results_samples.append(samples)

    return pd.DataFrame({
        "time_period": results_time,
        "location": results_location,
        "samples": results_samples,
    })


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

app = create_cli_app(train, predict, name="example-multistep-model")

if __name__ == "__main__":
    app()
