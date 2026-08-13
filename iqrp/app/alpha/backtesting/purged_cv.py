"""Purged K-Fold splits for overlapping-label time series.

Look-ahead / leakage prevention
-------------------------------
Observations within ``purge`` bars of the test fold are removed from the
training set so that labels whose horizon overlaps the test window cannot
leak into training. This is a local reimplementation of the purged-kfold
pattern (no dependency on tree_models).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def purged_kfold_splits(
    n: int,
    *,
    n_splits: int = 5,
    purge: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield purged train/test index pairs.

    For each contiguous test fold, train excludes indices in
    ``[test_start - purge, test_end + purge]``.
    """
    n = int(n)
    k = max(int(n_splits), 2)
    purge = max(int(purge), 0)
    fold = max(n // k, 1)
    for i in range(k):
        te_start = i * fold
        te_end = n if i == k - 1 else min((i + 1) * fold, n)
        if te_start >= te_end:
            continue
        te = np.arange(te_start, te_end)
        lo = max(te_start - purge, 0)
        hi = min(te_end - 1 + purge, n - 1)
        tr = np.array([j for j in range(n) if j < lo or j > hi], dtype=int)
        if tr.size and te.size:
            yield tr, te


def purge_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    purge: int = 5,
    n: int | None = None,
) -> np.ndarray:
    """Remove train indices within ``purge`` of any test index."""
    purge = max(int(purge), 0)
    if test_idx.size == 0:
        return np.asarray(train_idx, dtype=int)
    te0 = int(np.min(test_idx))
    te1 = int(np.max(test_idx))
    lo = te0 - purge
    hi = te1 + purge
    tr = np.asarray(train_idx, dtype=int)
    return tr[(tr < lo) | (tr > hi)]
