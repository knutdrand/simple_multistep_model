"""One-step probabilistic models for use with MultistepModel.

Contains:
- ResidualBootstrapModel: wraps any sklearn regressor, captures residuals for sampling
- BucketedResidualBootstrapModel: like above, but pops a 'bucket_id' feature
  column from X and pools residuals by it. Gated behind USE_RESIDUAL_BUCKETING.
- SkproWrapper: wraps a skpro probabilistic regressor
- Protocols: Distribution, OneStepModel

All one-step models accept ``X`` as an ``xr.DataArray`` with a ``feature``
dim and a feature-name coord. Wrappers strip the feature coord (and any
sentinel columns like ``bucket_id``) before handing values to sklearn /
skpro.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

import numpy as np
import xarray as xr

BUCKET_ID_FEATURE = "bucket_id"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class Distribution(Protocol):
    """Protocol for a probability distribution that supports sampling."""

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples. Returns shape (n_samples, n_rows)."""
        ...


class OneStepModel(Protocol):
    """Protocol for a one-step probabilistic regression model."""

    def fit(self, X: xr.DataArray, y: np.ndarray) -> None:
        """Fit on a (sample, feature) DataArray and (n_samples,) targets."""
        ...

    def predict_proba(self, X: xr.DataArray) -> Distribution:
        """Return a Distribution where sample(n) returns shape (n, n_rows)."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_bucket_ids(X: xr.DataArray) -> tuple[xr.DataArray, np.ndarray | None]:
    """Pop the bucket_id feature column if present.

    Returns (X_without_bucket, bucket_ids) where bucket_ids is an int array
    aligned with the sample dim, or None if the feature column wasn't there.
    """
    feature_names = X.coords["feature"].values
    if BUCKET_ID_FEATURE not in feature_names:
        return X, None
    bucket_ids = X.sel(feature=BUCKET_ID_FEATURE).values.astype(np.int64)
    keep = [f for f in feature_names if f != BUCKET_ID_FEATURE]
    return X.sel(feature=keep), bucket_ids


def _row_values(X: xr.DataArray) -> np.ndarray:
    """Return X as a (n_rows, n_features) float numpy array, regardless of dim order."""
    sample_dim = next(d for d in X.dims if d != "feature")
    return X.transpose(sample_dim, "feature").values


# ---------------------------------------------------------------------------
# ResidualDistribution + ResidualBootstrapModel  (no bucketing)
# ---------------------------------------------------------------------------


class ResidualDistribution:
    """Point predictions plus resampled residuals."""

    def __init__(self, predictions: np.ndarray, residuals: np.ndarray) -> None:
        self._predictions = predictions
        self._residuals = residuals

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples by adding resampled residuals to predictions.

        Returns shape (n_samples, n_rows), clamped to >= 0.
        """
        rng = np.random.default_rng()
        n_rows = len(self._predictions)
        drawn = rng.choice(self._residuals, size=(n_samples, n_rows), replace=True)
        samples = self._predictions[np.newaxis, :] + drawn
        return np.maximum(samples, 0.0)


class ResidualBootstrapModel:
    """One-step model wrapping any sklearn regressor with residual bootstrapping.

    Usage:
        model = ResidualBootstrapModel(GradientBoostingRegressor(n_estimators=100))
        model.fit(X_train, y_train)
        dist = model.predict_proba(X_test)
        samples = dist.sample(200)  # shape (200, n_test)
    """

    def __init__(self, regressor) -> None:
        self._regressor = regressor
        self._residuals: np.ndarray = np.array([0.0])

    def fit(self, X: xr.DataArray, y: np.ndarray) -> None:
        """Fit regressor and store training residuals."""
        values = _row_values(X)
        self._regressor.fit(values, y)
        self._residuals = y - self._regressor.predict(values)

    def predict_proba(self, X: xr.DataArray) -> ResidualDistribution:
        predictions = self._regressor.predict(_row_values(X))
        return ResidualDistribution(predictions, self._residuals)


# ---------------------------------------------------------------------------
# BucketedResidualDistribution + BucketedResidualBootstrapModel
# ---------------------------------------------------------------------------


class BucketedResidualDistribution:
    """Point predictions plus per-row residual pools (already bucket-resolved)."""

    def __init__(
        self,
        predictions: np.ndarray,
        pools_by_row: list[np.ndarray],
    ) -> None:
        if len(pools_by_row) != len(predictions):
            raise ValueError(
                "pools_by_row length must match predictions "
                f"({len(pools_by_row)} != {len(predictions)})"
            )
        self._predictions = predictions
        self._pools_by_row = pools_by_row

    def sample(self, n_samples: int) -> np.ndarray:
        rng = np.random.default_rng()
        n_rows = len(self._predictions)
        drawn = np.empty((n_samples, n_rows), dtype=float)
        for row_idx, pool in enumerate(self._pools_by_row):
            drawn[:, row_idx] = rng.choice(pool, size=n_samples, replace=True)
        samples = self._predictions[np.newaxis, :] + drawn
        return np.maximum(samples, 0.0)


class BucketedResidualBootstrapModel:
    """ResidualBootstrapModel variant that pools residuals by an int bucket id.

    Reads bucket ids from a sentinel ``bucket_id`` feature column on X,
    pops that column before handing the rest to the regressor, and groups
    residuals by id at fit time. At predict time it pre-binds each row's
    residual pool (falling back to the global pool when a row's bucket id
    has no fitted residuals).
    """

    def __init__(self, regressor) -> None:
        self._regressor = regressor
        self._global_residuals: np.ndarray = np.array([0.0])
        self._residuals_by_bucket: dict[int, np.ndarray] = {}

    def fit(self, X: xr.DataArray, y: np.ndarray) -> None:
        X_features, bucket_ids = _split_bucket_ids(X)
        values = _row_values(X_features)
        self._regressor.fit(values, y)
        residuals = y - self._regressor.predict(values)
        self._global_residuals = residuals
        self._residuals_by_bucket = {}

        if bucket_ids is None:
            return

        grouped: dict[int, list[float]] = defaultdict(list)
        for residual, bucket_id in zip(residuals, bucket_ids, strict=True):
            grouped[int(bucket_id)].append(float(residual))
        self._residuals_by_bucket = {
            key: np.asarray(values, dtype=float) for key, values in grouped.items()
        }

    def _pool(self, bucket_id: int) -> np.ndarray:
        pool = self._residuals_by_bucket.get(int(bucket_id))
        if pool is not None and len(pool) > 0:
            return pool
        return self._global_residuals

    def predict_proba(self, X: xr.DataArray) -> BucketedResidualDistribution:
        X_features, bucket_ids = _split_bucket_ids(X)
        predictions = self._regressor.predict(_row_values(X_features))
        if bucket_ids is None:
            pools = [self._global_residuals] * len(predictions)
        else:
            pools = [self._pool(int(b)) for b in bucket_ids]
        return BucketedResidualDistribution(predictions, pools)


# ---------------------------------------------------------------------------
# SkproWrapper
# ---------------------------------------------------------------------------


class SkproDistribution:
    """Wraps a skpro distribution object to conform to the Distribution protocol."""

    def __init__(self, skpro_dist) -> None:
        self._dist = skpro_dist

    def sample(self, n_samples: int) -> np.ndarray:
        # skpro's .sample(n) returns a DataFrame with MultiIndex.
        samples_df = self._dist.sample(n_samples)
        n_rows = len(self._dist)
        return samples_df.values.reshape(n_samples, n_rows)


class SkproWrapper:
    """Wraps a skpro probabilistic regressor to conform to the OneStepModel protocol.

    The only real job is reshaping skpro's MultiIndex DataFrame samples
    into the (n_samples, n_rows) numpy array that MultistepModel expects.
    """

    def __init__(self, skpro_model) -> None:
        self._model = skpro_model

    def fit(self, X: xr.DataArray, y: np.ndarray) -> None:
        self._model.fit(_row_values(X), y)

    def predict_proba(self, X: xr.DataArray) -> SkproDistribution:
        skpro_dist = self._model.predict_proba(_row_values(X))
        return SkproDistribution(skpro_dist)
