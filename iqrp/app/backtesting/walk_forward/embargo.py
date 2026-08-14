"""Embargo gaps after test folds to block serial leakage.

After a test fold ends, an embargo of ``embargo`` bars is excluded from
training so autocorrelation cannot bleed label information into the next
(or alternate) training sample.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from iqrp.app.backtesting.walk_forward.purge import purge_range


def embargo_range(test_end: int, *, embargo: int) -> tuple[int, int]:
    """Half-open ``[test_end, test_end + embargo)`` embargo zone."""
    e = max(int(embargo), 0)
    te = int(test_end)
    return te, te + e


def apply_embargo(
    train_idx: np.ndarray | Iterable[int],
    test_idx: np.ndarray | Iterable[int],
    *,
    embargo: int = 0,
    purge: int = 0,
) -> np.ndarray:
    """Drop train indices in the purge/embargo neighbourhood of the test fold."""
    tr = np.asarray(
        list(train_idx) if not isinstance(train_idx, np.ndarray) else train_idx, dtype=int
    )
    te = np.asarray(list(test_idx) if not isinstance(test_idx, np.ndarray) else test_idx, dtype=int)
    if tr.size == 0 or te.size == 0:
        return tr
    te0 = int(np.min(te))
    te1 = int(np.max(te)) + 1
    p = max(int(purge), 0)
    e = max(int(embargo), 0)
    lo = te0 - p
    hi = te1 + p + e
    return tr[(tr < lo) | (tr >= hi)]


def embargo_after_test(
    train_idx: np.ndarray | Iterable[int],
    *,
    test_end: int,
    embargo: int = 0,
) -> np.ndarray:
    """Remove only the post-test embargo zone from ``train_idx``."""
    tr = np.asarray(
        list(train_idx) if not isinstance(train_idx, np.ndarray) else train_idx, dtype=int
    )
    lo, hi = embargo_range(test_end, embargo=embargo)
    if hi <= lo:
        return tr
    return tr[(tr < lo) | (tr >= hi)]


def embargo_splits(
    n: int,
    *,
    n_splits: int = 5,
    embargo: int = 5,
    purge: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold with an additional post-test embargo window."""
    n = int(n)
    k = max(int(n_splits), 2)
    e = max(int(embargo), 0)
    p = max(int(purge), 0)
    fold = max(n // k, 1)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        te_start = i * fold
        te_end = n if i == k - 1 else min((i + 1) * fold, n)
        if te_start >= te_end:
            continue
        te = np.arange(te_start, te_end, dtype=int)
        lo, _ = purge_range(te_start, te_end, purge=p)
        hi = te_end + p + e
        tr = np.array([j for j in range(n) if j < lo or j >= hi], dtype=int)
        if tr.size and te.size:
            out.append((tr, te))
    return out
