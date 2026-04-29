"""Maps (location, time_value) -> bucket_id for residual bootstrapping.

The calculator follows a fit/transform pattern: at fit time it infers the
period granularity, counts rows per fine-grained ``(location, period)``
key, and freezes a mapping that collapses sparse buckets to coarser ones
(per-location, then global). At transform time it is a pure lookup —
no granularity re-inference, no fallback logic at the call site.

Bucket ids are strings:
    "{location}|{period_token}"   fine
    "{location}"                  per-location fallback
    "_global_"                    global fallback
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

GLOBAL_BUCKET = "_global_"


def _infer_granularity(times: Sequence) -> str:
    """Return 'week' if median delta <= 10 days, else 'month'."""
    if len(times) < 2:
        return "month"
    sorted_times = sorted(times)
    deltas = np.diff(np.array(sorted_times, dtype="datetime64[D]")).astype(int)
    if len(deltas) == 0:
        return "month"
    return "week" if int(np.median(deltas)) <= 10 else "month"


def _period_token(time_value, granularity: str) -> str:
    """Return MM or Wnn token for a datetime-like value or 'YYYY-Wnn' string.

    Weekly strings like 'YYYY-Wnn' are pandas-unfriendly, so we extract the
    'Wnn' suffix directly. Everything else is normalized via pandas.
    """
    if isinstance(time_value, str) and "W" in time_value:
        suffix = time_value.strip().rsplit("-", 1)[-1]
        if suffix.startswith("W"):
            return suffix
    ts = pd.Timestamp(time_value)
    if granularity == "week":
        return f"W{int(ts.isocalendar().week):02d}"
    return f"{int(ts.month):02d}"


def _resolve(
    location: str,
    period_token: str,
    fine_count: int,
    location_count: int,
    min_bucket_size: int,
) -> str:
    """Pick the coarsest bucket id whose pool clears min_bucket_size."""
    if fine_count >= min_bucket_size:
        return f"{location}|{period_token}"
    if location_count >= min_bucket_size:
        return location
    return GLOBAL_BUCKET


class BucketCalculator:
    """Maps (location, time_value) -> bucket_id, fallback frozen at fit time."""

    def __init__(self, min_bucket_size: int = 5) -> None:
        self._min_bucket_size = min_bucket_size
        self._granularity: str = "month"
        self._fine_to_id: dict[tuple[str, str], str] = {}
        self._location_counts: Counter[str] = Counter()

    def fit(self, locations: Sequence, times: Sequence) -> "BucketCalculator":
        """Build the (location, period_token) -> bucket_id map from training data."""
        if len(locations) != len(times):
            raise ValueError(
                f"locations and times must align ({len(locations)} != {len(times)})"
            )

        self._granularity = _infer_granularity(list(times))

        fine_keys = [
            (str(loc), _period_token(t, self._granularity))
            for loc, t in zip(locations, times, strict=True)
        ]
        fine_counts: Counter[tuple[str, str]] = Counter(fine_keys)
        self._location_counts = Counter(loc for loc, _ in fine_keys)

        self._fine_to_id = {
            (loc, tok): _resolve(
                loc, tok, fine_counts[(loc, tok)],
                self._location_counts[loc], self._min_bucket_size,
            )
            for (loc, tok) in fine_counts
        }
        return self

    def _bucket_for(self, location: str, period_token: str) -> str:
        cached = self._fine_to_id.get((location, period_token))
        if cached is not None:
            return cached
        # Unseen at fit time: fall through location -> global.
        if self._location_counts.get(location, 0) >= self._min_bucket_size:
            return location
        return GLOBAL_BUCKET

    def transform(self, locations: Sequence, times: Sequence) -> list[str]:
        """Look up the bucket id for each (location, time) row."""
        return [
            self._bucket_for(str(loc), _period_token(t, self._granularity))
            for loc, t in zip(locations, times, strict=True)
        ]

    def transform_one(self, location, time_value) -> str:
        """Look up the bucket id for a single (location, time) pair."""
        return self._bucket_for(str(location), _period_token(time_value, self._granularity))
