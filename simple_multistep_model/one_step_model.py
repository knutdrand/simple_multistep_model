"""One-step probabilistic models for use with MultistepModel.

Contains:
- ResidualBootstrapModel: wraps any sklearn regressor, captures residuals for sampling
- BucketedResidualBootstrapModel: like above, but with residuals pooled by an
  opaque bucket id (see ``bucket_calculator.BucketCalculator``). Gated behind
  USE_RESIDUAL_BUCKETING — see config.py.
- SkproWrapper: wraps a skpro probabilistic regressor
- Protocols: Distribution, OneStepModel
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class Distribution(Protocol):
    """Protocol for a probability distribution that supports sampling."""

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples.

        Returns:
            Shape (n_samples, n_rows).
        """
        ...


class OneStepModel(Protocol):
    """Protocol for a one-step probabilistic regression model."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on (n_samples, n_features) features and (n_samples,) targets."""
        ...

    def predict_proba(self, X: np.ndarray) -> Distribution:
        """Return a Distribution over next-step values.

        Args:
            X: Feature matrix, shape (n_rows, n_features).

        Returns:
            Distribution where sample(n) returns shape (n, n_rows).
        """
        ...


# ---------------------------------------------------------------------------
# ResidualDistribution + ResidualBootstrapModel  (main path)
# ---------------------------------------------------------------------------


class ResidualDistribution:
    """Point predictions plus resampled residuals."""

    def __init__(self, predictions: np.ndarray, residuals: np.ndarray) -> None:
        """Store predictions and training residuals.

        Args:
            predictions: Point predictions, shape (n_rows,).
            residuals: Training residuals for resampling, shape (n_train,).
        """
        self._predictions = predictions
        self._residuals = residuals

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples by adding resampled residuals to predictions.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Shape (n_samples, n_rows), clamped to >= 0.
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
        """Create from an sklearn regressor instance.

        Args:
            regressor: Any sklearn-compatible regressor with .fit() and .predict().
        """
        self._regressor = regressor
        self._residuals: np.ndarray = np.array([0.0])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit regressor and store training residuals."""
        self._regressor.fit(X, y)
        predictions = self._regressor.predict(X)
        self._residuals = y - predictions

    def predict_proba(self, X: np.ndarray) -> ResidualDistribution:
        """Return a ResidualDistribution over next-step values."""
        predictions = self._regressor.predict(X)
        return ResidualDistribution(predictions, self._residuals)


# ---------------------------------------------------------------------------
# BucketedResidualDistribution + BucketedResidualBootstrapModel  (new path)
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
        """Draw n_samples by adding resampled residuals from each row's pool.

        Returns shape (n_samples, n_rows), clamped to >= 0.
        """
        rng = np.random.default_rng()
        n_rows = len(self._predictions)
        drawn = np.empty((n_samples, n_rows), dtype=float)
        for row_idx, pool in enumerate(self._pools_by_row):
            drawn[:, row_idx] = rng.choice(pool, size=n_samples, replace=True)
        samples = self._predictions[np.newaxis, :] + drawn
        return np.maximum(samples, 0.0)


class BucketedResidualBootstrapModel:
    """ResidualBootstrapModel variant with residuals pooled by opaque bucket id.

    Bucket assignment (and any fallback) is the caller's responsibility — see
    ``bucket_calculator.BucketCalculator``. This class just stores residuals
    keyed by bucket id and returns the matching pool per row at predict time.

    Usage:
        model = BucketedResidualBootstrapModel(GradientBoostingRegressor(...))
        model.fit(X_train, y_train, bucket_ids=[...])
        dist = model.predict_proba(X_test, bucket_ids=[...])
        samples = dist.sample(200)
    """

    def __init__(self, regressor) -> None:
        self._regressor = regressor
        self._global_residuals: np.ndarray = np.array([0.0])
        self._residuals_by_bucket: dict[str, np.ndarray] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bucket_ids: list[str] | None = None,
    ) -> None:
        """Fit regressor; store residuals globally and grouped by bucket id."""
        self._regressor.fit(X, y)
        predictions = self._regressor.predict(X)
        residuals = y - predictions
        self._global_residuals = residuals
        self._residuals_by_bucket = {}

        if bucket_ids is None:
            return
        if len(bucket_ids) != len(residuals):
            raise ValueError(
                "bucket_ids length must match fitted samples "
                f"({len(bucket_ids)} != {len(residuals)})"
            )

        grouped: dict[str, list[float]] = defaultdict(list)
        for residual, bucket_id in zip(residuals, bucket_ids, strict=True):
            grouped[str(bucket_id)].append(float(residual))
        self._residuals_by_bucket = {
            key: np.asarray(values, dtype=float) for key, values in grouped.items()
        }

    def _pool(self, bucket_id: str) -> np.ndarray:
        pool = self._residuals_by_bucket.get(bucket_id)
        if pool is not None and len(pool) > 0:
            return pool
        return self._global_residuals

    def predict_proba(
        self,
        X: np.ndarray,
        bucket_ids: list[str] | None = None,
    ) -> BucketedResidualDistribution:
        """Return a distribution with residual pools pre-bound per row."""
        predictions = self._regressor.predict(X)
        if bucket_ids is None:
            pools = [self._global_residuals] * len(predictions)
        else:
            if len(bucket_ids) != len(predictions):
                raise ValueError(
                    "bucket_ids length must match prediction rows "
                    f"({len(bucket_ids)} != {len(predictions)})"
                )
            pools = [self._pool(str(b)) for b in bucket_ids]
        return BucketedResidualDistribution(predictions, pools)


# ---------------------------------------------------------------------------
# SkproWrapper
# ---------------------------------------------------------------------------


class SkproDistribution:
    """Wraps a skpro distribution object to conform to the Distribution protocol."""

    def __init__(self, skpro_dist) -> None:
        """Store the skpro distribution.

        Args:
            skpro_dist: A skpro distribution object (from predict_proba).
        """
        self._dist = skpro_dist

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples from the skpro distribution.

        Returns:
            Shape (n_samples, n_rows).
        """
        # skpro's .sample(n) returns a DataFrame with MultiIndex
        samples_df = self._dist.sample(n_samples)
        # Reshape to (n_samples, n_rows)
        n_rows = len(self._dist)
        return samples_df.values.reshape(n_samples, n_rows)


class SkproWrapper:
    """Wraps a skpro probabilistic regressor to conform to the OneStepModel protocol.

    The only real job is reshaping skpro's MultiIndex DataFrame samples
    into the (n_samples, n_rows) numpy array that MultistepModel expects.

    Usage:
        from skpro.regression.residual import ResidualDouble
        skpro_model = ResidualDouble(GradientBoostingRegressor())
        wrapper = SkproWrapper(skpro_model)
        wrapper.fit(X_train, y_train)
        dist = wrapper.predict_proba(X_test)
        samples = dist.sample(200)
    """

    def __init__(self, skpro_model) -> None:
        """Create from a skpro probabilistic regressor instance.

        Args:
            skpro_model: Any skpro regressor with .fit() and .predict_proba().
        """
        self._model = skpro_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the skpro model."""
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> SkproDistribution:
        """Return a SkproDistribution wrapping the skpro prediction."""
        skpro_dist = self._model.predict_proba(X)
        return SkproDistribution(skpro_dist)
