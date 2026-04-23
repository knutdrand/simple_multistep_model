"""Train a multistep forecasting model for disease prediction."""

import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from skpro.regression.bootstrap import BootstrapRegressor
from transformations import transform_data, add_lagged_targets
from simple_multistep_model import DataFrameMultistepModel, FixedMapieCrossConformalRegressor, SkproWrapper

N_TARGET_LAGS = 6
N_SAMPLES = 100
TARGET_VARIABLE = "disease_cases"
FEATURE_COLUMNS = ["rainfall", "mean_temperature", "mean_relative_humidity"]


def choose_regressor(raw_X: pd.DataFrame, y: pd.Series):
    X = add_lagged_targets(raw_X, y[TARGET_VARIABLE], max_lag=N_TARGET_LAGS)
    na_mask = ~(X.isna().any(axis=1) | y[TARGET_VARIABLE].isna()).values
    X, y = X[na_mask], y[na_mask]
    groups = X["location"]
    X = X.drop(columns=['time_period', 'location'])
    y = y[TARGET_VARIABLE]

    # Custom scorers (use RMSE as primary)
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    cv = GroupKFold(n_splits=5)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_dist = {
        "n_estimators": randint(100, 1001),           # 100–1000 trees
        "max_depth": randint(6, 41),                  # 6–40 (None can overfit; try explicit depths)
        "min_samples_split": randint(2, 21),          # 2–20
        "min_samples_leaf": randint(1, 11),           # 1–10
        "max_features": ["sqrt", "log2", 0.5, 0.7],   # common good options
        "bootstrap": [True]                           # RF default & usually best
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,                 # increase if you want a deeper search
        scoring=rmse_scorer,       # primary metric: (negative) RMSE on log scale
        refit=True,                # refit best on full data
        cv=cv.split(X, y, groups=groups),
        verbose=1,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )

    search.fit(X, y)
    best_rf = search.best_estimator_
    print("Best params:", search.best_params_)
    print("Best CV RMSE (log scale):", -search.best_score_)
    return best_rf

def train(train_data_path: str, model_path: str) -> None:
    data = pd.read_csv(train_data_path)

    index_cols = ["time_period", "location"]
    y = data[index_cols + [TARGET_VARIABLE]]
    y[TARGET_VARIABLE] = np.log1p(y[TARGET_VARIABLE])
    X = data[index_cols + FEATURE_COLUMNS]
    X = transform_data(X)
    regressor = choose_regressor(X, y)
    #regressor = RandomForestRegressor(max_depth=10, min_samples_leaf=5, max_features="sqrt", )
    # prob_cls = ResidualDouble
    # prob_cls = MapieCrossConformalRegressor
    prob_cls = BootstrapRegressor
    skpro_model = prob_cls(regressor)
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
