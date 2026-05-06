"""Train a multistep forecasting model for disease prediction."""

import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from skpro.regression.bootstrap import BootstrapRegressor

from transformations import transform_data, add_lagged_targets
from simple_multistep_model import (
    BucketCalculator,
    BucketedResidualBootstrapModel,
    DataFrameMultistepModel,
    FixedMapieCrossConformalRegressor,
    RunConfig,
    SkproWrapper,
    load_run_config,
)

INDEX_COLS = ["time_period", "location"]


def choose_regressor(
    raw_X: pd.DataFrame,
    y: pd.DataFrame,
    target_variable: str,
    n_target_lags: int,
):
    """Tune a RandomForestRegressor via RandomizedSearchCV with GroupKFold by location."""
    X = add_lagged_targets(raw_X, y[target_variable], max_lag=n_target_lags)
    na_mask = ~(X.isna().any(axis=1) | y[target_variable].isna()).values
    X, y = X[na_mask], y[na_mask]
    groups = X["location"]
    X = X.drop(columns=["time_period", "location"])
    y = y[target_variable]

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    cv = GroupKFold(n_splits=5)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": randint(100, 1001),
        "max_depth": randint(6, 41),
        "min_samples_split": randint(2, 21),
        "min_samples_leaf": randint(1, 11),
        "max_features": ["sqrt", "log2", 0.5, 0.7],
        "bootstrap": [True],
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        scoring=rmse_scorer,
        refit=True,
        cv=cv.split(X, y, groups=groups),
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )

    search.fit(X, y)
    print("Best params:", search.best_params_)
    print("Best CV RMSE (log scale):", -search.best_score_)
    return search.best_estimator_


def train(train_data_path: str, model_path: str, config_path: str | None = None) -> None:
    cfg = load_run_config(config_path) if config_path else RunConfig()

    data = pd.read_csv(train_data_path)
    y = data[INDEX_COLS + [cfg.target_variable]]
    X = data[INDEX_COLS + cfg.feature_columns]
    X = transform_data(X, min_lag=cfg.feature_min_lag, max_lag=cfg.feature_max_lag)

    if cfg.tune_regressor:
        # choose_regressor must search in the same target space the model
        # will actually fit in; the multistep model log1p's internally, so
        # mirror that here when tuning.
        y_for_search = y.copy()
        if cfg.log_transform_target:
            y_for_search[cfg.target_variable] = np.log1p(y_for_search[cfg.target_variable])
        regressor = choose_regressor(X, y_for_search, cfg.target_variable, cfg.n_target_lags)
    else:
        regressor = RandomForestRegressor(
            n_estimators=cfg.rf.n_estimators,
            max_depth=cfg.rf.max_depth,
            min_samples_leaf=cfg.rf.min_samples_leaf,
            max_features=cfg.rf.max_features,
            random_state=cfg.rf.random_state,
        )

    bucket_calculator = None
    if cfg.prob_wrapper == "bucketedresidual":
        one_step = BucketedResidualBootstrapModel(regressor)
        bucket_calculator = BucketCalculator(min_bucket_size=cfg.min_bucket_size)
    elif cfg.prob_wrapper == "bootstrap":
        one_step = SkproWrapper(BootstrapRegressor(regressor))
    elif cfg.prob_wrapper == "cross-conformal":
        one_step = SkproWrapper(FixedMapieCrossConformalRegressor(regressor))

    model = DataFrameMultistepModel(
        one_step,
        cfg.n_target_lags,
        cfg.target_variable,
        bucket_calculator=bucket_calculator,
        log_transform_target=cfg.log_transform_target,
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
