"""Pydantic config for one train/predict run.

A run config is normally loaded from a YAML file and threaded through both
`train.py` and `predict.py` so that knobs (feature columns, lag depth,
sample count, regressor params, residual-bucketing toggle, ...) live in
one place rather than scattered as module-level constants.

Example YAML::

    target_variable: disease_cases
    feature_columns: [rainfall, mean_temperature]
    n_target_lags: 6
    n_samples: 100
    use_residual_bucketing: false
    rf:
      n_estimators: 100
      max_depth: 10
      min_samples_leaf: 5
      max_features: sqrt
      random_state: 42
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RandomForestConfig(BaseModel):
    """Hyperparameters for the underlying sklearn RandomForestRegressor."""

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = 100
    max_depth: int | None = 10
    min_samples_leaf: int = 5
    max_features: str | int | float | None = "sqrt"
    random_state: int | None = None


class RunConfig(BaseModel):
    """All tunable knobs for a single train/predict run."""

    model_config = ConfigDict(extra="forbid")

    target_variable: str = "disease_cases"
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "rainfall",
            "mean_temperature",
            "mean_relative_humidity",
        ]
    )

    n_target_lags: int = 6
    n_samples: int = 100

    feature_min_lag: int = 1
    feature_max_lag: int = 3

    use_residual_bucketing: bool = False
    min_bucket_size: int = 5

    tune_regressor: bool = False

    rf: RandomForestConfig = Field(default_factory=RandomForestConfig)


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a RunConfig from a YAML file.

    The file must exist and contain a YAML mapping with valid RunConfig
    fields. An empty `{}` mapping is fine and yields all defaults.
    """
    with Path(path).open("r") as f:
        return RunConfig.model_validate(yaml.safe_load(f))
