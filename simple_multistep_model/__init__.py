"""Simple multistep recursive forecasting model.

Compose these pieces to build a forecasting pipeline:
1. An sklearn regressor (any regressor)
2. A probabilistic wrapper (ResidualBootstrapModel or SkproWrapper)
3. A MultistepModel for recursive forecasting
4. A data-transform function (plain DataFrame -> DataFrame)
5. A CLI via create_cli_app
"""

from simple_multistep_model.bucket_calculator import BucketCalculator
from simple_multistep_model.run_config import (
    RandomForestConfig,
    RunConfig,
    load_run_config,
)
from simple_multistep_model.multistep import (
    DataFrameMultistepModel,
    DeterministicMultistepModel,
    MultistepModel,
    MultistepDistribution,
    target_to_xarray,
    features_to_xarray,
    future_features_to_xarray,
)
from simple_multistep_model.one_step_model import (
    BucketedResidualBootstrapModel,
    BucketedResidualDistribution,
    FixedMapieCrossConformalRegressor,
    ResidualBootstrapModel,
    ResidualDistribution,
    SkproWrapper,
)
from simple_multistep_model.cli import create_cli_app

__all__ = [
    "BucketCalculator",
    "DataFrameMultistepModel",
    "DeterministicMultistepModel",
    "MultistepModel",
    "MultistepDistribution",
    "BucketedResidualBootstrapModel",
    "BucketedResidualDistribution",
    "FixedMapieCrossConformalRegressor",
    "ResidualBootstrapModel",
    "ResidualDistribution",
    "SkproWrapper",
    "RandomForestConfig",
    "RunConfig",
    "create_cli_app",
    "load_run_config",
    "target_to_xarray",
    "features_to_xarray",
    "future_features_to_xarray",
]
