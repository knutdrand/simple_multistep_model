"""Example: composing pieces into a working multistep forecasting CLI.

Shows how to wire together:
1. A plain data-transform function (DataFrame -> DataFrame)
2. An sklearn regressor
3. A bootstrap wrapper (ResidualBootstrapModel)
4. The DataFrameMultistepModel for recursive forecasting
5. A CLI for train/predict from CSV

Run:
    python -m simple_multistep_model.example train-cmd --help
    python -m simple_multistep_model.example train-cmd train.csv model.pkl
    python -m simple_multistep_model.example predict-cmd model.pkl historic.csv future.csv predictions.csv
"""

from __future__ import annotations

import pandas as pd
import xarray as xr
from sklearn.ensemble import GradientBoostingRegressor

from simple_multistep_model.cli import create_cli_app
from simple_multistep_model.multistep import DataFrameMultistepModel
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


def train(data: pd.DataFrame) -> DataFrameMultistepModel:
    """Train a multistep model from a training DataFrame."""
    data = transform_data(data)

    index_cols = ["time_period", "location"]
    feature_cols = [c for c in data.columns if c not in index_cols + [TARGET_VARIABLE]]

    y = data[index_cols + [TARGET_VARIABLE]]
    X = data[index_cols + feature_cols] if feature_cols else None

    sklearn_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    one_step = ResidualBootstrapModel(sklearn_model)
    model = DataFrameMultistepModel(one_step, N_TARGET_LAGS, TARGET_VARIABLE)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# 3. Predict function
# ---------------------------------------------------------------------------


def predict(model: DataFrameMultistepModel, historic: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions from trained model."""
    historic = transform_data(historic)
    future = transform_data(future)

    index_cols = ["time_period", "location"]
    feature_cols = [c for c in future.columns if c not in index_cols + [TARGET_VARIABLE]]

    y_historic = historic[index_cols + [TARGET_VARIABLE]]
    X_future = future[index_cols + feature_cols] if feature_cols else None

    n_steps = future.groupby("location").size().iloc[0]
    predictions = model.predict(y_historic, X_future, n_steps, N_SAMPLES)

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
