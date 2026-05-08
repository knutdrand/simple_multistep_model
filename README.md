# simple_multistep_model

A multistep recursive forecasting model for disease prediction, designed to plug into the [CHAP](https://github.com/dhis2-chap/chap-core) framework as an external model. Internally it composes an sklearn regressor (`RandomForestRegressor` by default) with a probabilistic wrapper (`bootstrap`, `cross-conformal`, or `bucketedresidual`) and recursive lag-feature handling.

## Installation

This project uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

## Configuration

Both `chap eval` (and `chap forecast`) and the standalone `train.py` / `predict.py` scripts read a single YAML file in the shape chap calls a *model configuration*. The wrapper has two top-level keys:

| Key | What it holds |
|-----|---------------|
| `additional_continuous_covariates` | List of column names in the dataset to use as model features. |
| `user_option_values` | Model-specific tuning knobs (lag depth, sample count, regressor params, ...). |

This mirrors `chap_core.database.model_templates_and_config_tables.ModelConfiguration`. The pydantic schema lives in [`simple_multistep_model/run_config.py`](simple_multistep_model/run_config.py) — see `ChapModelConfiguration` for the wrapper and `RunConfig` for the inner option set.

### Example

```yaml
additional_continuous_covariates:
  - rainfall
  - mean_temperature
  - mean_relative_humidity
user_option_values:
  target_variable: disease_cases   # column in the dataset to forecast
  n_target_lags: 6                  # lagged target values fed back as features
  n_samples: 100                    # number of probabilistic samples per prediction
  feature_min_lag: 1                # smallest feature lag (in periods)
  feature_max_lag: 3                # largest feature lag (in periods)
  prob_wrapper: bootstrap           # bucketedresidual | bootstrap | cross-conformal
  log_transform_target: true        # fit on log1p(target) and clamp predictions at 0
  tune_regressor: false             # if true, RandomizedSearchCV picks RF hyperparams
  min_bucket_size: 5                # only consulted when prob_wrapper=bucketedresidual
  rf:                               # RandomForestRegressor hyperparameters
    n_estimators: 100
    max_depth: 10
    min_samples_leaf: 5
    max_features: sqrt
    random_state: 42
```

Both objects use pydantic with `extra="forbid"`, so unknown keys are rejected — that includes the legacy bare-`RunConfig` shape (e.g. putting `feature_columns` directly under `user_option_values` no longer works; covariate columns belong at the top level).

## Running with chap eval

Point chap at the model directory and pass the wrapped YAML via `--model-configuration-yaml`:

```bash
chap eval \
  --model-name /path/to/simple_multistep_model \
  --dataset-csv data/dataset.csv \
  --output-file output/output.nc \
  --model-configuration-yaml config.yaml \
  --backtest-params.n-periods 6 \
  --backtest-params.n-splits 7 \
  --backtest-params.stride 1
```

chap validates `config.yaml` against its `ModelConfiguration` schema and writes a normalized copy as `runs/<timestamp>/model_configuration_for_run.yaml`. The model's `train.py` / `predict.py` read that file via `load_model_configuration` and unpack:

- `wrapper.user_option_values` — a `RunConfig` with the model knobs
- `wrapper.additional_continuous_covariates` — the list of feature column names

The dataset CSV must have `time_period`, `location`, the column named in `target_variable`, and every column listed in `additional_continuous_covariates`.

## Running standalone

The same scripts also work outside chap. They take a config path via `--config`:

```bash
# Train
python train.py training_data.csv model.pkl --config config.yaml

# Predict
python predict.py model.pkl historic.csv future.csv predictions.csv --config config.yaml
```

The config file must be in the same wrapped shape described above.
