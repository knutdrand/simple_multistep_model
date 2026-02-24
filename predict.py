"""Generate predictions using a trained multistep model."""

import argparse
import pickle

import pandas as pd

from simple_multistep_model import DataFrameMultistepModel
from transformations import transform_data

INDEX_COLS = ["time_period", "location"]
N_SAMPLES = 200
TARGET_VARIABLE = "disease_cases"
FEATURE_COLUMNS = ["rainfall", "mean_temperature", "mean_relative_humidity"]
X_COLUMNS = INDEX_COLS + FEATURE_COLUMNS



def predict(
    model_path: str,
    historic_data_path: str,
    future_data_path: str,
    out_file_path: str,
) -> None:
    with open(model_path, "rb") as f:
        model: DataFrameMultistepModel = pickle.load(f)

    historic = pd.read_csv(historic_data_path)
    future = pd.read_csv(future_data_path)
    n_steps = future.groupby("location").size().iloc[0]
    features = pd.concat([historic[X_COLUMNS], future[X_COLUMNS]], ignore_index=True)
    features = features.sort_values(by=["time_period", 'location'])
    X = transform_data(features)
    y_historic = historic[INDEX_COLS + [TARGET_VARIABLE]]
    predictions = model.predict(y_historic, X, n_steps, N_SAMPLES)

    predictions.to_csv(out_file_path, index=False)
    print(f"Predictions saved to {out_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate disease predictions")
    parser.add_argument("model", help="Path to trained model file")
    parser.add_argument("historic_data", help="Path to historic data CSV file")
    parser.add_argument("future_data", help="Path to future climate data CSV file")
    parser.add_argument("out_file", help="Path to save predictions CSV file")
    args = parser.parse_args()

    predict(args.model, args.historic_data, args.future_data, args.out_file)
