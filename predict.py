"""Generate predictions using a trained multistep model."""

import argparse
import pickle

import pandas as pd

from simple_multistep_model import DataFrameMultistepModel, load_run_config
from transformations import transform_data

INDEX_COLS = ["time_period", "location"]


def predict(
    model_path: str,
    historic_data_path: str,
    future_data_path: str,
    out_file_path: str,
    config_path: str | None = None,
) -> None:
    cfg = load_run_config(config_path)

    with open(model_path, "rb") as f:
        model: DataFrameMultistepModel = pickle.load(f)

    historic = pd.read_csv(historic_data_path)
    future = pd.read_csv(future_data_path)
    n_steps = future.groupby("location").size().iloc[0]
    x_columns = INDEX_COLS + cfg.feature_columns
    features = pd.concat([historic[x_columns], future[x_columns]], ignore_index=True)
    features = features.sort_values(by=["time_period", "location"])
    X = transform_data(features, min_lag=cfg.feature_min_lag, max_lag=cfg.feature_max_lag)
    y_historic = historic[INDEX_COLS + [cfg.target_variable]]
    predictions = model.predict(y_historic, X, n_steps, cfg.n_samples)

    predictions.to_csv(out_file_path, index=False)
    print(f"Predictions saved to {out_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate disease predictions")
    parser.add_argument("model", help="Path to trained model file")
    parser.add_argument("historic_data", help="Path to historic data CSV file")
    parser.add_argument("future_data", help="Path to future climate data CSV file")
    parser.add_argument("out_file", help="Path to save predictions CSV file")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a RunConfig YAML file (defaults are used if omitted)",
    )
    args = parser.parse_args()

    predict(args.model, args.historic_data, args.future_data, args.out_file, args.config)
