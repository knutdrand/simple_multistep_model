"""Train a multistep forecasting model for disease prediction."""

import argparse
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skpro.regression.residual import ResidualDouble
from transformations import transform_data
from simple_multistep_model import (
    BucketedResidualBootstrapModel,
    DataFrameMultistepModel,
    RunConfig,
    SkproWrapper,
    load_run_config,
)

INDEX_COLS = ["time_period", "location"]


def train(train_data_path: str, model_path: str, config_path: str | None = None) -> None:
    cfg = load_run_config(config_path) if config_path else RunConfig()

    data = pd.read_csv(train_data_path)
    y = data[INDEX_COLS + [cfg.target_variable]]
    X = data[INDEX_COLS + cfg.feature_columns]
    X = transform_data(X, min_lag=cfg.feature_min_lag, max_lag=cfg.feature_max_lag)

    regressor = RandomForestRegressor(
        n_estimators=cfg.rf.n_estimators,
        max_depth=cfg.rf.max_depth,
        min_samples_leaf=cfg.rf.min_samples_leaf,
        max_features=cfg.rf.max_features,
        random_state=cfg.rf.random_state,
    )
    if cfg.use_residual_bucketing:
        one_step = BucketedResidualBootstrapModel(regressor, min_bucket_size=cfg.min_bucket_size)
    else:
        skpro_model = ResidualDouble(regressor)
        one_step = SkproWrapper(skpro_model)

    model = DataFrameMultistepModel(
        one_step,
        cfg.n_target_lags,
        cfg.target_variable,
        use_residual_bucketing=cfg.use_residual_bucketing,
    )
    model.fit(X, y)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multistep disease prediction model")
    parser.add_argument("train_data", help="Path to training data CSV file")
    parser.add_argument("model", help="Path to save the trained model")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a RunConfig YAML file (defaults are used if omitted)",
    )
    args = parser.parse_args()

    train(args.train_data, args.model, args.config)
