"""Pydantic config for one train/predict run.

A run config is normally loaded from a YAML file and threaded through both
`train.py` and `predict.py` so that knobs (feature columns, lag depth,
sample count, regressor params, probabilistic-wrapper choice, ...) live in
one place rather than scattered as module-level constants.

Example YAML::

    target_variable: disease_cases
    feature_columns: [rainfall, mean_temperature]
    n_target_lags: 6
    n_samples: 100
    prob_wrapper: bootstrap
    rf:
      n_estimators: 100
      max_depth: 10
      min_samples_leaf: 5
      max_features: sqrt
      random_state: 42
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ProbWrapper = Literal["bucketedresidual", "bootstrap", "cross-conformal"]


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

    prob_wrapper: ProbWrapper = "bootstrap"
    min_bucket_size: int = 5

    log_transform_target: bool = True
    tune_regressor: bool = False

    rf: RandomForestConfig = Field(default_factory=RandomForestConfig)


class ChapModelConfiguration(BaseModel):
    """Wrapper YAML that ``chap eval``/``chap forecast`` hands to the model.

    The user passes their config to chap via ``--model-configuration-yaml``
    already in this shape: model-specific knobs under ``user_option_values``
    and any extra covariate columns the data carries listed under
    ``additional_continuous_covariates``. chap validates that file, then
    writes it through to ``model_configuration_for_run.yaml`` in the run
    directory for ``train``/``predict`` to read. Mirrors
    ``chap_core.database.model_templates_and_config_tables.ModelConfiguration``.
    """

    model_config = ConfigDict(extra="forbid")

    additional_continuous_covariates: list[str] = Field(default_factory=list)
    user_option_values: RunConfig = Field(default_factory=RunConfig)


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a RunConfig from a chap model-configuration YAML.

    The file must contain a YAML mapping in the shape chap writes as
    ``model_configuration_for_run.yaml`` (see :class:`ChapModelConfiguration`):
    the user's options live under ``user_option_values`` and any extra
    covariate names under ``additional_continuous_covariates``. An empty
    ``{}`` mapping is fine and yields all defaults.
    """
    with Path(path).open("r") as f:
        wrapper = ChapModelConfiguration.model_validate(yaml.safe_load(f))
    return wrapper.user_option_values
