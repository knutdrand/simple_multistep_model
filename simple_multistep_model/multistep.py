"""Multistep recursive forecasting models.

Contains:
- MultistepModel: probabilistic recursive forecaster (uses OneStepModel protocol)
- DeterministicMultistepModel: point-prediction recursive forecaster
- MultistepDistribution: lazy distribution for recursive trajectory sampling
- Lag matrix builders and xarray conversion helpers
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Protocols (duplicated from one_step_model for self-containment)
# ---------------------------------------------------------------------------


class Distribution(Protocol):
    """Protocol for a probability distribution that supports sampling."""

    def sample(self, n_samples: int) -> np.ndarray:
        """Returns shape (n_samples, n_rows)."""
        ...


class OneStepModel(Protocol):
    """Protocol for a one-step probabilistic regression model."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict_proba(self, X: np.ndarray) -> Distribution: ...


class DeterministicOneStepModel(Protocol):
    """Protocol for a one-step deterministic regression model (e.g. sklearn regressor)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Lag matrix builders
# ---------------------------------------------------------------------------


def _build_lag_matrix_xr(y: xr.DataArray, n_lags: int) -> xr.DataArray:
    """Build lag matrix from DataArray with a time dim.

    Args:
        y: DataArray with at least a 'time' dim. May also have 'location'.
        n_lags: Number of lags.

    Returns:
        DataArray with an added 'lag' dim, time trimmed by n_lags.
        Lag order: oldest to newest [y(t-n_lags), ..., y(t-1)].
    """
    shifted = [y.shift(time=k) for k in range(n_lags, 0, -1)]
    lag_matrix = xr.concat(shifted, dim="lag")
    return lag_matrix.isel(time=slice(n_lags, None))


def _build_lag_matrix(y: np.ndarray, n_lags: int) -> xr.DataArray:
    """Build a lag matrix from a 1-d time series.

    Returns a DataArray with dims (time, lag), shape (len(y) - n_lags, n_lags).
    Columns ordered oldest to newest: [y(t-n_lags), ..., y(t-1)].
    """
    n = len(y) - n_lags
    cols = [y[i : i + n] for i in range(n_lags)]
    return xr.DataArray(np.column_stack(cols), dims=["time", "lag"])


# ---------------------------------------------------------------------------
# xarray conversion helpers
# ---------------------------------------------------------------------------


def target_to_xarray(y_df, target_variable: str = "disease_cases") -> xr.DataArray:
    """Pivot target DataFrame to xr.DataArray (location, time).

    Args:
        y_df: DataFrame with columns [time_period, location, <target_variable>].
        target_variable: Name of the target column.
    """
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
    """Pivot features DataFrame to xr.DataArray (location, time, feature) or None.

    Args:
        X_df: DataFrame with columns [time_period, location, feat1, feat2, ...].
    """
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
        coords={"location": locations, "time": times},
    )


def future_features_to_xarray(X_df) -> xr.DataArray | None:
    """Pivot future features DataFrame to xr.DataArray (location, step, feature) or None.

    Args:
        X_df: DataFrame with columns [time_period, location, feat1, ...].
    """
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
        coords={"location": locations},
    )


# ---------------------------------------------------------------------------
# MultistepDistribution
# ---------------------------------------------------------------------------


class MultistepDistribution:
    """Lazy distribution that runs recursive trajectory sampling on .sample()."""

    def __init__(
        self,
        model: OneStepModel,
        previous_y: np.ndarray,
        n_steps: int,
        n_target_lags: int,
        X: np.ndarray | None,
    ):
        self._model = model
        self._previous_y = previous_y
        self._n_steps = n_steps
        self._n_target_lags = n_target_lags
        self._X = X

    def sample(self, n: int) -> np.ndarray:
        """Generate n recursive trajectories.

        Returns shape (n, n_steps). Each row is one sampled trajectory.
        """
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
                )
                features = xr.concat([exog, lag_window.rename(lag="feature")], dim="feature")
            else:
                features = lag_window.rename(lag="feature")

            dist = self._model.predict_proba(features.values)
            step_samples = xr.DataArray(dist.sample(1)[0], dims=["trajectory"])
            step_results.append(step_samples)

            lag_window = lag_window.roll(lag=-1)
            lag_window[{"lag": -1}] = step_samples

        trajectories = xr.concat(step_results, dim="step")
        return trajectories.transpose("trajectory", "step").values


# ---------------------------------------------------------------------------
# MultistepModel (probabilistic)
# ---------------------------------------------------------------------------


class MultistepModel:
    """Wraps a OneStepModel to produce multi-step recursive forecasts with uncertainty."""

    def __init__(self, one_step_model: OneStepModel, n_target_lags: int):
        self.one_step_model = one_step_model
        self.n_target_lags = n_target_lags

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> None:
        """Build lag matrix from y, append to X, and train the one-step model.

        Args:
            y: Target time series, shape (n_timepoints,).
            X: Exogenous features, shape (n_timepoints, n_features) or None.
        """
        lags = _build_lag_matrix(y, self.n_target_lags)
        y_target = y[self.n_target_lags :]

        if X is not None:
            exog = xr.DataArray(X[self.n_target_lags :], dims=["time", "feature"])
            features = xr.concat([exog, lags.rename(lag="feature")], dim="feature")
        else:
            features = lags.rename(lag="feature")

        self.one_step_model.fit(features.values, y_target)

    def fit_multi(self, y: xr.DataArray, X: xr.DataArray | None = None) -> None:
        """Fit on multi-location data, pooling all locations into one training set.

        Args:
            y: Target values, dims (location, time).
            X: Exogenous features, dims (location, time, feature) or None.
        """
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

        self.one_step_model.fit(
            features_stacked.transpose("sample", "feature").values,
            y_stacked.values,
        )

    def predict_proba(
        self,
        previous_y: np.ndarray,
        n_steps: int,
        X: np.ndarray | None = None,
    ) -> MultistepDistribution:
        """Return a lazy MultistepDistribution for recursive sampling.

        Args:
            previous_y: Most recent observed values, shape (n_target_lags,), newest last.
            n_steps: Number of steps to forecast.
            X: Known future exogenous features, shape (n_steps, n_features) or None.
        """
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
    ) -> xr.DataArray:
        """Generate multi-step forecasts for multiple locations.

        Args:
            previous_y: Recent observations, dims (location, time), >= n_target_lags timepoints.
            n_steps: Number of forecast steps.
            n_samples: Number of sampled trajectories per location.
            X: Known future exogenous features, dims (location, step, feature) or None.

        Returns:
            DataArray with dims (location, trajectory, step).
        """
        locations = previous_y.coords["location"].values
        results = []
        for loc in locations:
            prev = previous_y.sel(location=loc).values
            X_loc = X.sel(location=loc).values if X is not None else None
            dist = self.predict_proba(prev, n_steps, X_loc)
            samples = dist.sample(n_samples)
            results.append(samples)

        return xr.DataArray(
            np.stack(results),
            dims=["location", "trajectory", "step"],
            coords={"location": locations},
        )


# ---------------------------------------------------------------------------
# DeterministicMultistepModel (point predictions only)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DataFrameMultistepModel (thin pandas wrapper)
# ---------------------------------------------------------------------------


class DataFrameMultistepModel:
    """Thin pandas wrapper around MultistepModel.

    Handles DataFrame <-> xarray conversion so the user works
    entirely in pandas. No target transforms — do those in your
    data-transform function.
    """

    def __init__(
        self,
        one_step_model: OneStepModel,
        n_target_lags: int,
        target_variable: str = "disease_cases",
    ) -> None:
        self._model = MultistepModel(one_step_model, n_target_lags)
        self._target_variable = target_variable

    @property
    def n_target_lags(self) -> int:
        return self._model.n_target_lags

    def fit(self, X, y) -> None:
        """Fit on feature and target DataFrames (multi-location).

        Args:
            X: DataFrame with [time_period, location, feat1, ...] or None.
            y: DataFrame with [time_period, location, <target_variable>].
        """
        y_xr = target_to_xarray(y, self._target_variable)
        X_xr = features_to_xarray(X) if X is not None else None
        self._model.fit_multi(y_xr, X_xr)

    def predict(self, y_historic, X_future, n_steps: int, n_samples: int):
        """Predict from DataFrames, return xarray (location, trajectory, step).

        Args:
            y_historic: DataFrame with [time_period, location, <target_variable>].
            X_future: DataFrame with [time_period, location, feat1, ...] or None.
            n_steps: Number of forecast steps.
            n_samples: Number of sampled trajectories.
        """
        y_xr = target_to_xarray(y_historic, self._target_variable)
        previous_y = y_xr.isel(time=slice(-self._model.n_target_lags, None))
        X_xr = future_features_to_xarray(X_future) if X_future is not None else None
        return self._model.predict_multi(previous_y, n_steps, n_samples, X_xr)


class DeterministicMultistepModel:
    """Recursive multi-step forecaster using point predictions only (no sampling).

    Each step feeds the point prediction forward as input to the next step.
    Supports multi-location pooling via fit_multi/predict_multi.
    """

    def __init__(self, one_step_model: DeterministicOneStepModel, n_target_lags: int):
        self.one_step_model = one_step_model
        self.n_target_lags = n_target_lags

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> None:
        """Build lag matrix from y, append to X, and train the one-step model."""
        lags = _build_lag_matrix(y, self.n_target_lags)
        y_target = y[self.n_target_lags :]

        if X is not None:
            exog = xr.DataArray(X[self.n_target_lags :], dims=["time", "feature"])
            features = xr.concat([exog, lags.rename(lag="feature")], dim="feature")
        else:
            features = lags.rename(lag="feature")

        self.one_step_model.fit(features.values, y_target)

    def fit_multi(self, y: xr.DataArray, X: xr.DataArray | None = None) -> None:
        """Fit on multi-location data, pooling all locations."""
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

        self.one_step_model.fit(
            features_stacked.transpose("sample", "feature").values,
            y_stacked.values,
        )

    def predict(
        self,
        previous_y: np.ndarray,
        n_steps: int,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate deterministic multi-step forecast.

        Args:
            previous_y: Recent observations, shape (>= n_target_lags,).
            n_steps: Number of forecast steps.
            X: Known future exogenous features, shape (n_steps, n_features) or None.

        Returns:
            Array of shape (n_steps,) with point predictions.
        """
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
        """Generate deterministic multi-step forecasts for multiple locations.

        Returns:
            DataArray with dims (location, step).
        """
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
