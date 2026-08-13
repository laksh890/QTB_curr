"""Purge overlapping labels between train and test windows.

Observations whose forward-looking label horizon overlaps the test fold are
removed from the training set to prevent leakage (Lopez de Prado purged CV).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def purge_range(test_start: int, test_end: int, *, purge: int) -> tuple[int, int]:
    """Inclusive-style neighbourhood around the test fold that must leave train.

    Returns half-open ``[lo, hi)`` covering
    ``[test_start - purge, test_end + purge)``.
    """
    p = max(int(purge), 0)
    lo = int(test_start) - p
    hi = int(test_end) + p
    return lo, hi


def purge_train_indices(
    train_idx: np.ndarray | Iterable[int],
    test_idx: np.ndarray | Iterable[int],
    *,
    purge: int = 0,
) -> np.ndarray:
    """Remove train indices within ``purge`` bars of any test index."""
    tr = np.asarray(list(train_idx) if not isinstance(train_idx, np.ndarray) else train_idx, dtype=int)
    te = np.asarray(list(test_idx) if not isinstance(test_idx, np.ndarray) else test_idx, dtype=int)
    if tr.size == 0 or te.size == 0:
        return tr
    p = max(int(purge), 0)
    if p == 0:
        # Still drop exact overlap.
        te_set = set(int(x) for x in te.tolist())
        return tr[np.array([int(i) not in te_set for i in tr], dtype=bool)] if te_set else tr
    te0 = int(np.min(te))
    te1 = int(np.max(te)) + 1  # exclusive end of test span
    lo, hi = purge_range(te0, te1, purge=p)
    return tr[(tr < lo) | (tr >= hi)]


def apply_purge(
    train_idx: np.ndarray | Iterable[int],
    *,
    test_start: int,
    test_end: int,
    purge: int = 0,
) -> np.ndarray:
    """Purge train indices overlapping ``[test_start, test_end)`` by ``purge``."""
    te = np.arange(int(test_start), int(test_end), dtype=int)
    return purge_train_indices(train_idx, te, purge=purge)


def purged_kfold_splits(
    n: int,
    *,
    n_splits: int = 5,
    purge: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous purged K-fold splits over ``[0, n)``.

    For each test fold, train excludes ``[test_start - purge, test_end + purge)``.
    """
    n = int(n)
    k = max(int(n_splits), 2)
    p = max(int(purge), 0)
    fold = max(n // k, 1)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        te_start = i * fold
        te_end = n if i == k - 1 else min((i + 1) * fold, n)
        if te_start >= te_end:
            continue
        te = np.arange(te_start, te_end, dtype=int)
        lo, hi = purge_range(te_start, te_end, purge=p)
        tr = np.array([j for j in range(n) if j < lo or j >= hi], dtype=int)
        if tr.size and te.size:
            out.append((tr, te))
    return out
