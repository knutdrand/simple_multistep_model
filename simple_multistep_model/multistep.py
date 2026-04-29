"""Multistep recursive forecasting models.

Contains:
- MultistepModel: probabilistic recursive forecaster (uses OneStepModel protocol).
  Optionally owns a BucketCalculator that annotates the feature DataArray
  with a ``bucket_id`` column at fit and predict time.
- DeterministicMultistepModel: point-prediction recursive forecaster
- MultistepDistribution: lazy distribution running recursive trajectory sampling
- Lag matrix builders and xarray conversion helpers
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import xarray as xr

from simple_multistep_model.bucket_calculator import BucketCalculator
from simple_multistep_model.one_step_model import BUCKET_ID_FEATURE


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class Distribution(Protocol):
    """Protocol for a probability distribution that supports sampling."""

    def sample(self, n_samples: int) -> np.ndarray:
        """Returns shape (n_samples, n_rows)."""
        ...


class OneStepModel(Protocol):
    """Protocol for a one-step probabilistic regression model."""

    def fit(self, X: xr.DataArray, y: np.ndarray) -> None: ...

    def predict_proba(self, X: xr.DataArray) -> Distribution: ...


class DeterministicOneStepModel(Protocol):
    """Protocol for a one-step deterministic regression model (e.g. sklearn regressor)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Lag matrix builders
# ---------------------------------------------------------------------------


def _lag_feature_names(n_lags: int) -> list[str]:
    """Names for lag columns, oldest first: ['lag_n', ..., 'lag_1']."""
    return [f"lag_{n_lags - i}" for i in range(n_lags)]


def _build_lag_matrix_xr(y: xr.DataArray, n_lags: int) -> xr.DataArray:
    """Build lag matrix from DataArray with a time dim.

    Returns a DataArray with an added 'lag' dim, time trimmed by n_lags.
    Lag order: oldest to newest [y(t-n_lags), ..., y(t-1)].
    """
    shifted = [y.shift(time=k) for k in range(n_lags, 0, -1)]
    lag_matrix = xr.concat(shifted, dim="lag")
    return lag_matrix.isel(time=slice(n_lags, None))


def _build_lag_matrix(y: np.ndarray, n_lags: int) -> xr.DataArray:
    """Build a lag matrix from a 1-d time series (oldest to newest)."""
    n = len(y) - n_lags
    cols = [y[i : i + n] for i in range(n_lags)]
    return xr.DataArray(np.column_stack(cols), dims=["time", "lag"])


def _lags_as_features(lags: xr.DataArray, n_lags: int) -> xr.DataArray:
    """Rename the 'lag' dim to 'feature' and attach human-readable names."""
    return lags.rename(lag="feature").assign_coords(feature=_lag_feature_names(n_lags))


# ---------------------------------------------------------------------------
# bucket_id annotation
# ---------------------------------------------------------------------------


def _add_bucket_id_column(features: xr.DataArray, bucket_ids: np.ndarray) -> xr.DataArray:
    """Append a ``bucket_id`` feature column to a DataArray with a 'feature' dim."""
    other_dims = [d for d in features.dims if d != "feature"]
    shape = (1,) + tuple(features.sizes[d] for d in other_dims)
    bucket_da = xr.DataArray(
        np.asarray(bucket_ids, dtype=np.int64).reshape(shape),
        dims=["feature"] + other_dims,
        coords={"feature": [BUCKET_ID_FEATURE]},
    )
    return xr.concat([features, bucket_da], dim="feature")


# ---------------------------------------------------------------------------
# xarray conversion helpers
# ---------------------------------------------------------------------------


def target_to_xarray(y_df, target_variable: str = "disease_cases") -> xr.DataArray:
    """Pivot target DataFrame to xr.DataArray (location, time)."""
    import pandas as pd

    df = y_df.copy()
    df["time_period"] = pd.to_datetime(df["time_period"])
    target_wide = df.pivot(index="time_period", columns="location", values=target_variable)
    target_wide = target_wide.sort_index().ffill().bfill()
    locations = list(target_wide.columns)
    times = list(target_wide.index)
    return xr.DataArray(
        target_wide.values.T,
        dims=["location", "time"],
        coords={"location": locations, "time": times},
    )


def features_to_xarray(X_df) -> xr.DataArray | None:
    """Pivot features DataFrame to xr.DataArray (location, time, feature) with feature names."""
    import pandas as pd

    index_cols = ["time_period", "location"]
    feature_cols = [c for c in X_df.columns if c not in index_cols]
    if not feature_cols:
        return None

    df = X_df.copy()
    df["time_period"] = pd.to_datetime(df["time_period"])

    feature_arrays = []
    for var in feature_cols:
        var_wide = df.pivot(index="time_period", columns="location", values=var)
        var_wide = var_wide.sort_index().ffill().bfill()
        feature_arrays.append(var_wide.values.T)

    locations = sorted(df["location"].unique().tolist())
    times = sorted(df["time_period"].unique().tolist())

    return xr.DataArray(
        np.stack(feature_arrays, axis=-1),
        dims=["location", "time", "feature"],
        coords={"location": locations, "time": times, "feature": feature_cols},
    )


def future_features_to_xarray(X_df) -> xr.DataArray | None:
    """Pivot future features DataFrame to xr.DataArray (location, step, feature)."""
    import pandas as pd

    index_cols = ["time_period", "location"]
    feature_cols = [c for c in X_df.columns if c not in index_cols]
    if not feature_cols:
        return None

    df = X_df.copy()
    df["time_period"] = pd.to_datetime(df["time_period"])
    locations = sorted(df["location"].unique().tolist())

    feature_arrays = []
    for var in feature_cols:
        var_wide = df.pivot(index="time_period", columns="location", values=var)
        var_wide = var_wide.sort_index().ffill().bfill()
        var_wide = var_wide[locations]
        feature_arrays.append(var_wide.values.T)

    return xr.DataArray(
        np.stack(feature_arrays, axis=-1),
        dims=["location", "step", "feature"],
        coords={"location": locations, "feature": feature_cols},
    )


def _predictions_to_dataframe(predictions: xr.DataArray, future_df=None):
    """Convert xarray predictions (location, trajectory, step) to wide DataFrame."""
    import pandas as pd

    locations = predictions.coords["location"].values

    if future_df is not None:
        df = future_df.copy()
        original_times = df["time_period"].astype(str)
        df["_original_time"] = original_times
        df["time_period"] = pd.to_datetime(df["time_period"])
        time_lookup: dict[str, list[str]] = {}
        for loc in locations:
            loc_str = str(loc)
            loc_subset = df[df["location"] == loc_str].sort_values("time_period")
            time_lookup[loc_str] = loc_subset["_original_time"].values.tolist()
    else:
        time_lookup = None

    rows = []
    for loc in locations:
        loc_str = str(loc)
        loc_preds = predictions.sel(location=loc)
        n_steps = loc_preds.sizes["step"]

        for step_idx in range(n_steps):
            samples = loc_preds.isel(step=step_idx).values
            time_val = time_lookup[loc_str][step_idx] if time_lookup else step_idx
            row = {"time_period": time_val, "location": loc_str}
            for i, s in enumerate(samples):
                row[f"sample_{i}"] = s
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MultistepDistribution
# ---------------------------------------------------------------------------


class MultistepDistribution:
    """Lazy distribution that runs recursive trajectory sampling on .sample().

    When ``step_bucket_ids`` is provided, each step's feature DataArray gets
    a ``bucket_id`` column appended before being handed to the one-step
    model. The model is responsible for popping that column.
    """

    def __init__(
        self,
        model: OneStepModel,
        previous_y: np.ndarray,
        n_steps: int,
        n_target_lags: int,
        X: np.ndarray | None,
        feature_names: list[str] | None = None,
        step_bucket_ids: list[int] | None = None,
    ):
        self._model = model
        self._previous_y = previous_y
        self._n_steps = n_steps
        self._n_target_lags = n_target_lags
        self._X = X
        self._feature_names = feature_names or []
        self._step_bucket_ids = step_bucket_ids

    def sample(self, n: int) -> np.ndarray:
        """Generate n recursive trajectories, shape (n, n_steps)."""
        lag_names = _lag_feature_names(self._n_target_lags)
        lag_window = xr.DataArray(
            np.tile(self._previous_y, (n, 1)),
            dims=["trajectory", "lag"],
        )

        step_results: list[xr.DataArray] = []
        for step in range(self._n_steps):
            if self._X is not None:
                exog = xr.DataArray(
                    np.tile(self._X[step], (n, 1)),
                    dims=["trajectory", "feature"],
                    coords={"feature": self._feature_names},
                )
                features = xr.concat(
                    [exog, lag_window.rename(lag="feature").assign_coords(feature=lag_names)],
                    dim="feature",
                )
            else:
                features = lag_window.rename(lag="feature").assign_coords(feature=lag_names)

            if self._step_bucket_ids is not None:
                bucket_id = self._step_bucket_ids[step]
                features = _add_bucket_id_column(features, np.full(n, bucket_id, dtype=np.int64))

            dist = self._model.predict_proba(features)
            step_samples = xr.DataArray(dist.sample(1)[0], dims=["trajectory"])
            step_results.append(step_samples)

            lag_window = lag_window.roll(lag=-1)
            lag_window[{"lag": -1}] = step_samples

        trajectories = xr.concat(step_results, dim="step")
        return trajectories.transpose("trajectory", "step").values


# ---------------------------------------------------------------------------
# MultistepModel
# ---------------------------------------------------------------------------


class MultistepModel:
    """Wraps a OneStepModel to produce multi-step recursive forecasts with uncertainty.

    Optionally takes a ``BucketCalculator``; when provided, fit_multi annotates
    the feature DataArray with a ``bucket_id`` column (the calculator is fitted
    on the training (location, time) index) and predict_multi annotates each
    forecast step using ``future_times``.
    """

    def __init__(
        self,
        one_step_model: OneStepModel,
        n_target_lags: int,
        bucket_calculator: BucketCalculator | None = None,
    ):
        self.one_step_model = one_step_model
        self.n_target_lags = n_target_lags
        self.bucket_calculator = bucket_calculator

    def _feature_names(self, X: xr.DataArray | None) -> list[str]:
        if X is None or "feature" not in X.coords:
            return []
        return [str(f) for f in X.coords["feature"].values]

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> None:
        """Build lag matrix from y, append to X, and train the one-step model."""
        lags = _build_lag_matrix(y, self.n_target_lags)
        y_target = y[self.n_target_lags :]
        lags_feat = _lags_as_features(lags, self.n_target_lags)

        if X is not None:
            exog_X = X[self.n_target_lags :]
            exog = xr.DataArray(
                exog_X,
                dims=["time", "feature"],
                coords={"feature": [f"x_{i}" for i in range(exog_X.shape[1])]},
            )
            features = xr.concat([exog, lags_feat], dim="feature")
        else:
            features = lags_feat

        values = features.values
        mask = ~(np.isnan(values).any(axis=1) | np.isnan(y_target))
        self.one_step_model.fit(features.isel(time=mask), y_target[mask])

    def fit_multi(self, y: xr.DataArray, X: xr.DataArray | None = None) -> None:
        """Fit on multi-location data, pooling all locations into one training set."""
        lags = _build_lag_matrix_xr(y, self.n_target_lags)
        y_target = y.isel(time=slice(self.n_target_lags, None))

        lags_feat = _lags_as_features(lags, self.n_target_lags)
        if X is not None:
            X_trimmed = X.isel(time=slice(self.n_target_lags, None))
            features = xr.concat(
                [X_trimmed.transpose("feature", "location", "time"), lags_feat],
                dim="feature",
            )
        else:
            features = lags_feat

        features_stacked = features.stack(sample=("location", "time")).transpose(
            "sample", "feature"
        )
        y_stacked = y_target.stack(sample=("location", "time"))

        values = features_stacked.values
        y_np = y_stacked.values
        mask = ~(np.isnan(values).any(axis=1) | np.isnan(y_np))

        if self.bucket_calculator is not None:
            sample_index = y_stacked.coords["sample"].to_index()
            kept = [pair for pair, keep in zip(sample_index, mask, strict=True) if keep]
            kept_locations = [str(loc) for loc, _ in kept]
            kept_times = [t for _, t in kept]
            self.bucket_calculator.fit(kept_locations, kept_times)
            bucket_ids = self.bucket_calculator.transform(kept_locations, kept_times)
            X_fit = _add_bucket_id_column(features_stacked.isel(sample=mask), bucket_ids)
        else:
            X_fit = features_stacked.isel(sample=mask)

        self.one_step_model.fit(X_fit, y_np[mask])

    def predict_proba(
        self,
        previous_y: np.ndarray,
        n_steps: int,
        X: np.ndarray | None = None,
    ) -> MultistepDistribution:
        """Return a lazy MultistepDistribution for recursive sampling."""
        return MultistepDistribution(
            model=self.one_step_model,
            previous_y=previous_y[-self.n_target_lags :],
            n_steps=n_steps,
            n_target_lags=self.n_target_lags,
            X=X,
        )

    def predict_multi(
        self,
        previous_y: xr.DataArray,
        n_steps: int,
        n_samples: int,
        X: xr.DataArray | None = None,
        future_times: dict[str, list] | None = None,
    ) -> xr.DataArray:
        """Generate multi-step forecasts for multiple locations.

        Args:
            future_times: Per-location list of time values for each forecast
                step. Required when this MultistepModel has a bucket_calculator.
        """
        locations = previous_y.coords["location"].values
        feature_names = self._feature_names(X)
        future_times = future_times or {}
        results = []
        for loc in locations:
            prev = previous_y.sel(location=loc).values
            X_loc = X.sel(location=loc).values if X is not None else None
            loc_str = str(loc)

            step_bucket_ids: list[int] | None = None
            if self.bucket_calculator is not None:
                times_for_loc = future_times.get(loc_str)
                if times_for_loc is None:
                    raise ValueError(
                        f"future_times[{loc_str!r}] required when bucket_calculator is set"
                    )
                if len(times_for_loc) != n_steps:
                    raise ValueError(
                        f"future_times[{loc_str!r}] has {len(times_for_loc)} entries, "
                        f"expected {n_steps}"
                    )
                step_bucket_ids = [
                    self.bucket_calculator.transform_one(loc_str, t)
                    for t in times_for_loc
                ]

            dist = MultistepDistribution(
                model=self.one_step_model,
                previous_y=prev[-self.n_target_lags :],
                n_steps=n_steps,
                n_target_lags=self.n_target_lags,
                X=X_loc,
                feature_names=feature_names,
                step_bucket_ids=step_bucket_ids,
            )
            samples = dist.sample(n_samples)
            results.append(samples)

        return xr.DataArray(
            np.stack(results),
            dims=["location", "trajectory", "step"],
            coords={"location": locations},
        )


# ---------------------------------------------------------------------------
# DataFrameMultistepModel (thin pandas wrapper)
# ---------------------------------------------------------------------------


class DataFrameMultistepModel:
    """Thin pandas wrapper around MultistepModel.

    Handles DataFrame <-> xarray conversion. Pass a ``BucketCalculator`` to
    pool residuals by (location, period); when omitted, no bucketing happens.
    """

    def __init__(
        self,
        one_step_model: OneStepModel,
        n_target_lags: int,
        target_variable: str = "disease_cases",
        bucket_calculator: BucketCalculator | None = None,
    ) -> None:
        self._model = MultistepModel(
            one_step_model, n_target_lags, bucket_calculator=bucket_calculator
        )
        self._target_variable = target_variable

    @property
    def n_target_lags(self) -> int:
        return self._model.n_target_lags

    def fit(self, X, y) -> None:
        """Fit on feature and target DataFrames (multi-location)."""
        y_xr = target_to_xarray(y, self._target_variable)
        X_xr = features_to_xarray(X) if X is not None else None
        self._model.fit_multi(y_xr, X_xr)

    def predict(self, y_historic, X, n_steps: int, n_samples: int):
        """Predict from DataFrames, return wide-format DataFrame."""
        y_xr = target_to_xarray(y_historic, self._target_variable)
        previous_y = y_xr.isel(time=slice(-self._model.n_target_lags, None))

        X_xr = features_to_xarray(X)
        X_future_xr = X_xr.isel(time=slice(-n_steps, None)).rename({"time": "step"})
        future_df = X.groupby("location", sort=False).tail(n_steps)

        future_times = (
            {
                str(loc): group["time_period"].astype(str).tolist()
                for loc, group in future_df.groupby("location", sort=False)
            }
            if self._model.bucket_calculator is not None
            else None
        )

        predictions = self._model.predict_multi(
            previous_y,
            n_steps,
            n_samples,
            X_future_xr,
            future_times=future_times,
        )

        return _predictions_to_dataframe(predictions, future_df)


class DeterministicMultistepModel:
    """Recursive multi-step forecaster using point predictions only (no sampling).

    Uses the simpler numpy-based ``DeterministicOneStepModel`` protocol — no
    xarray, no bucketing.
    """

    def __init__(self, one_step_model: DeterministicOneStepModel, n_target_lags: int):
        self.one_step_model = one_step_model
        self.n_target_lags = n_target_lags

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> None:
        lags = _build_lag_matrix(y, self.n_target_lags)
        y_target = y[self.n_target_lags :]

        if X is not None:
            exog = xr.DataArray(X[self.n_target_lags :], dims=["time", "feature"])
            features = xr.concat([exog, lags.rename(lag="feature")], dim="feature")
        else:
            features = lags.rename(lag="feature")

        X_np = features.values
        y_np = y_target
        mask = ~(np.isnan(X_np).any(axis=1) | np.isnan(y_np))
        self.one_step_model.fit(X_np[mask], y_np[mask])

    def fit_multi(self, y: xr.DataArray, X: xr.DataArray | None = None) -> None:
        lags = _build_lag_matrix_xr(y, self.n_target_lags)
        y_target = y.isel(time=slice(self.n_target_lags, None))

        lags_feat = lags.rename(lag="feature")
        if X is not None:
            X_trimmed = X.isel(time=slice(self.n_target_lags, None))
            features = xr.concat(
                [X_trimmed.transpose("feature", "location", "time"), lags_feat],
                dim="feature",
            )
        else:
            features = lags_feat

        features_stacked = features.stack(sample=("location", "time"))
        y_stacked = y_target.stack(sample=("location", "time"))

        X_np = features_stacked.transpose("sample", "feature").values
        y_np = y_stacked.values
        mask = ~(np.isnan(X_np).any(axis=1) | np.isnan(y_np))
        self.one_step_model.fit(X_np[mask], y_np[mask])

    def predict(
        self,
        previous_y: np.ndarray,
        n_steps: int,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        lag_window = previous_y[-self.n_target_lags :].copy().astype(float)
        results = []
        for step in range(n_steps):
            if X is not None:
                features = np.concatenate([X[step], lag_window]).reshape(1, -1)
            else:
                features = lag_window.reshape(1, -1)
            pred = float(self.one_step_model.predict(features)[0])
            results.append(pred)
            lag_window = np.roll(lag_window, -1)
            lag_window[-1] = pred
        return np.array(results)

    def predict_multi(
        self,
        previous_y: xr.DataArray,
        n_steps: int,
        X: xr.DataArray | None = None,
    ) -> xr.DataArray:
        locations = previous_y.coords["location"].values
        results = []
        for loc in locations:
            prev = previous_y.sel(location=loc).values
            X_loc = X.sel(location=loc).values if X is not None else None
            preds = self.predict(prev, n_steps, X_loc)
            results.append(preds)

        return xr.DataArray(
            np.stack(results),
            dims=["location", "step"],
            coords={"location": locations},
        )
