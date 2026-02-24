"""Train a multistep forecasting model for disease prediction."""

import argparse
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skpro.regression.residual import ResidualDouble
from transformations import transform_data
from simple_multistep_model import DataFrameMultistepModel, SkproWrapper

N_TARGET_LAGS = 6
N_SAMPLES = 100
TARGET_VARIABLE = "disease_cases"
FEATURE_COLUMNS = ["rainfall", "mean_temperature", "mean_relative_humidity"]




def train(train_data_path: str, model_path: str) -> None:
    data = pd.read_csv(train_data_path)

    index_cols = ["time_period", "location"]
    y = data[index_cols + [TARGET_VARIABLE]]
    X = data[index_cols + FEATURE_COLUMNS]
    X = transform_data(X)
    regressor = RandomForestRegressor(max_depth=10, min_samples_leaf=5, max_features="sqrt", )
    skpro_model = ResidualDouble(regressor)
    one_step = SkproWrapper(skpro_model)
    model = DataFrameMultistepModel(one_step, N_TARGET_LAGS, TARGET_VARIABLE)
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multistep disease prediction model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    args = parser.parse_args()

    train(args.train_data, args.model)
