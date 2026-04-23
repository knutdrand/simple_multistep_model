"""One-step probabilistic models for use with MultistepModel.

Contains:
- ResidualBootstrapModel: wraps any sklearn regressor, captures residuals for sampling
- SkproWrapper: wraps a skpro probabilistic regressor
- FixedMapieCrossConformalRegressor: bugfixed skpro subclass
- Protocols: Distribution, OneStepModel
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from skpro.regression.conformal import MapieCrossConformalRegressor


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
# ResidualDistribution + ResidualBootstrapModel
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
# SkproWrapper
# ---------------------------------------------------------------------------


class FixedMapieCrossConformalRegressor(MapieCrossConformalRegressor):
    """MapieCrossConformalRegressor with fixed predict_proba.

    skpro's default _predict_proba constructs Normal(mu, sigma) where mu is a
    1-D numpy array and sigma is a 2-D DataFrame.  The shape mismatch causes
    Normal.shape to broadcast incorrectly (e.g. (n, n) instead of (n, 1)),
    which crashes Normal.sample().

    This subclass ensures mu and sigma have consistent shapes.
    """

    def _predict_proba(self, X):
        from skpro.distributions.normal import Normal

        pred_mean = self.predict(X=X)
        pred_var = self.predict_var(X=X)
        pred_std = np.sqrt(pred_var)

        if hasattr(X, "index"):
            index = X.index
        else:
            index = pd.RangeIndex(start=0, stop=len(X), step=1)
        columns = self._get_columns(method="predict")

        # Coerce mu to a DataFrame matching sigma's shape to prevent
        # numpy broadcasting from inflating Normal.shape.
        if not isinstance(pred_mean, pd.DataFrame):
            pred_mean = pd.DataFrame(pred_mean, index=index, columns=columns)

        return Normal(mu=pred_mean, sigma=pred_std, index=index, columns=columns)


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
