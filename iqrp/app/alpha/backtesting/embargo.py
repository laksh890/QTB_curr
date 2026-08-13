"""Embargo gaps after test folds to block serial leakage.

Look-ahead prevention
---------------------
After a test fold ends, an embargo of ``embargo`` bars is also excluded from
training. Combined with purge before the fold, this blocks both backward and
forward label overlap around the test window.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    embargo: int = 5,
    purge: int = 0,
) -> np.ndarray:
    """Drop train indices in the purge/embargo neighbourhood of the test fold."""
    if test_idx.size == 0:
        return np.asarray(train_idx, dtype=int)
    te0 = int(np.min(test_idx))
    te1 = int(np.max(test_idx))
    lo = te0 - max(int(purge), 0)
    hi = te1 + max(int(purge), 0) + max(int(embargo), 0)
    tr = np.asarray(train_idx, dtype=int)
    return tr[(tr < lo) | (tr > hi)]


def embargo_splits(
    n: int,
    *,
    n_splits: int = 5,
    embargo: int = 5,
    purge: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold with an additional post-test embargo window."""
    n = int(n)
    k = max(int(n_splits), 2)
    embargo = max(int(embargo), 0)
    purge = max(int(purge), 0)
    fold = max(n // k, 1)
    for i in range(k):
        te_start = i * fold
        te_end = n if i == k - 1 else min((i + 1) * fold, n)
        if te_start >= te_end:
            continue
        te = np.arange(te_start, te_end)
        lo = max(te_start - purge, 0)
        hi = min(te_end - 1 + purge + embargo, n - 1)
        tr = np.array([j for j in range(n) if j < lo or j > hi], dtype=int)
        if tr.size and te.size:
            yield tr, te
