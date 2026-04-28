"""One-step probabilistic models for use with MultistepModel.

Contains:
- ResidualBootstrapModel: wraps any sklearn regressor, captures residuals for sampling
- BucketedResidualBootstrapModel: like above, but with per-location / per-period
  residual pools. Gated behind USE_RESIDUAL_BUCKETING — see config.py.
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
    """Point predictions plus resampled residuals, bucketed by (location, period).

    Falls back to per-location residuals, then to the global pool, when a
    bucket is smaller than ``min_bucket_size``.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        residuals_by_location: dict[str, np.ndarray],
        residuals_by_location_period: dict[tuple[str, str], np.ndarray],
        min_bucket_size: int,
    ) -> None:
        self._predictions = predictions
        self._residuals = residuals
        self._residuals_by_location = residuals_by_location
        self._residuals_by_location_period = residuals_by_location_period
        self._min_bucket_size = min_bucket_size

    def _pool_for_context(self, location: str, period_token: str) -> np.ndarray:
        period_pool = self._residuals_by_location_period.get((location, period_token))
        if period_pool is not None and len(period_pool) >= self._min_bucket_size:
            return period_pool

        location_pool = self._residuals_by_location.get(location)
        if location_pool is not None and len(location_pool) > 0:
            return location_pool

        return self._residuals

    def sample(
        self,
        n_samples: int,
        context_by_row: list[tuple[str, str]] | None = None,
    ) -> np.ndarray:
        """Draw samples by adding resampled residuals to predictions.

        Args:
            n_samples: Number of samples to draw.
            context_by_row: Optional list of (location, period_token) for each row.
                When provided, residuals are drawn from the matching bucket.

        Returns:
            Shape (n_samples, n_rows), clamped to >= 0.
        """
        rng = np.random.default_rng()
        n_rows = len(self._predictions)

        if context_by_row is None:
            drawn = rng.choice(self._residuals, size=(n_samples, n_rows), replace=True)
        else:
            if len(context_by_row) != n_rows:
                raise ValueError(
                    "context_by_row length must match number of prediction rows "
                    f"({len(context_by_row)} != {n_rows})"
                )
            drawn = np.empty((n_samples, n_rows), dtype=float)
            for row_idx, (location, period_token) in enumerate(context_by_row):
                pool = self._pool_for_context(str(location), str(period_token))
                drawn[:, row_idx] = rng.choice(pool, size=n_samples, replace=True)

        samples = self._predictions[np.newaxis, :] + drawn
        return np.maximum(samples, 0.0)


class BucketedResidualBootstrapModel:
    """ResidualBootstrapModel variant with per-location / per-period buckets.

    Usage:
        model = BucketedResidualBootstrapModel(GradientBoostingRegressor(...))
        model.fit(X_train, y_train, residual_context=[(loc, period), ...])
        dist = model.predict_proba(X_test)
        samples = dist.sample(200, context_by_row=[(loc, period), ...])
    """

    def __init__(self, regressor, min_bucket_size: int = 5) -> None:
        self._regressor = regressor
        self._residuals: np.ndarray = np.array([0.0])
        self._residuals_by_location: dict[str, np.ndarray] = {}
        self._residuals_by_location_period: dict[tuple[str, str], np.ndarray] = {}
        self._min_bucket_size = min_bucket_size

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        residual_context: list[tuple[str, str]] | None = None,
    ) -> None:
        """Fit regressor and store training residuals (globally and per bucket)."""
        self._regressor.fit(X, y)
        predictions = self._regressor.predict(X)
        self._residuals = y - predictions

        self._residuals_by_location = {}
        self._residuals_by_location_period = {}

        if residual_context is None:
            return

        if len(residual_context) != len(self._residuals):
            raise ValueError(
                "residual_context length must match fitted samples "
                f"({len(residual_context)} != {len(self._residuals)})"
            )

        by_location: dict[str, list[float]] = defaultdict(list)
        by_location_period: dict[tuple[str, str], list[float]] = defaultdict(list)
        for residual, (location, period_token) in zip(
            self._residuals, residual_context, strict=True
        ):
            loc = str(location)
            token = str(period_token)
            by_location[loc].append(float(residual))
            by_location_period[(loc, token)].append(float(residual))

        self._residuals_by_location = {
            loc: np.asarray(values, dtype=float) for loc, values in by_location.items()
        }
        self._residuals_by_location_period = {
            key: np.asarray(values, dtype=float)
            for key, values in by_location_period.items()
        }

    def predict_proba(self, X: np.ndarray) -> BucketedResidualDistribution:
        """Return a BucketedResidualDistribution over next-step values."""
        predictions = self._regressor.predict(X)
        return BucketedResidualDistribution(
            predictions,
            self._residuals,
            self._residuals_by_location,
            self._residuals_by_location_period,
            self._min_bucket_size,
        )


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
