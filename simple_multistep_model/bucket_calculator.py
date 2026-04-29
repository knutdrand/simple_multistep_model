"""Maps (location, time_value) -> int bucket_id for residual bootstrapping.

The calculator follows a fit/transform pattern: at fit time it infers the
period granularity, counts rows per fine-grained ``(location, period)``
key, and freezes a mapping that collapses sparse buckets to coarser ones
(per-location, then global). At transform time it is a pure lookup —
no granularity re-inference, no fallback logic at the call site.

Bucket ids are non-negative integers; the integer 0 is reserved for the
global pool. Use ``bucket_label(bucket_id)`` to recover a human-readable
form for diagnostics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

GLOBAL_BUCKET: int = 0
GLOBAL_LABEL: str = "_global_"


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


class BucketCalculator:
    """Maps (location, time_value) -> int bucket id, fallback frozen at fit time."""

    def __init__(self, min_bucket_size: int = 5) -> None:
        self._min_bucket_size = min_bucket_size
        self._granularity: str = "month"
        # (location, period_token) -> int id (resolved at fit time)
        self._fine_to_id: dict[tuple[str, str], int] = {}
        # location -> int id (used as fallback for unseen periods at predict time)
        self._location_to_id: dict[str, int] = {}
        # int id -> human-readable label, for diagnostics
        self._id_to_label: dict[int, str] = {GLOBAL_BUCKET: GLOBAL_LABEL}

    def fit(self, locations: Sequence, times: Sequence) -> "BucketCalculator":
        """Build the (location, period_token) -> int id map from training data."""
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
        location_counts: Counter[str] = Counter(loc for loc, _ in fine_keys)

        self._fine_to_id = {}
        self._location_to_id = {}
        self._id_to_label = {GLOBAL_BUCKET: GLOBAL_LABEL}
        next_id = GLOBAL_BUCKET + 1

        # Pre-assign per-location coarse ids for any location with enough rows.
        for loc, count in location_counts.items():
            if count >= self._min_bucket_size:
                self._location_to_id[loc] = next_id
                self._id_to_label[next_id] = loc
                next_id += 1

        for (loc, tok), count in fine_counts.items():
            if count >= self._min_bucket_size:
                self._fine_to_id[(loc, tok)] = next_id
                self._id_to_label[next_id] = f"{loc}|{tok}"
                next_id += 1
            elif loc in self._location_to_id:
                self._fine_to_id[(loc, tok)] = self._location_to_id[loc]
            else:
                self._fine_to_id[(loc, tok)] = GLOBAL_BUCKET

        return self

    def _bucket_for(self, location: str, period_token: str) -> int:
        cached = self._fine_to_id.get((location, period_token))
        if cached is not None:
            return cached
        # Unseen at fit time: fall through location -> global.
        return self._location_to_id.get(location, GLOBAL_BUCKET)

    def transform(self, locations: Sequence, times: Sequence) -> np.ndarray:
        """Look up bucket ids for each (location, time) row, returning an int array."""
        ids = [
            self._bucket_for(str(loc), _period_token(t, self._granularity))
            for loc, t in zip(locations, times, strict=True)
        ]
        return np.asarray(ids, dtype=np.int64)

    def transform_one(self, location, time_value) -> int:
        """Look up the bucket id for a single (location, time) pair."""
        return self._bucket_for(str(location), _period_token(time_value, self._granularity))

    def bucket_label(self, bucket_id: int) -> str:
        """Return a human-readable label for an int bucket id (for diagnostics)."""
        return self._id_to_label.get(int(bucket_id), GLOBAL_LABEL)
