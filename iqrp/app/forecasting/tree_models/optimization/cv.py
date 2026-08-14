"""Time-series aware cross-validation splits."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from iqrp.app.forecasting.tree_models.config import ValidationConfig


def make_time_splits(n: int, cfg: ValidationConfig | Any) -> list[tuple[np.ndarray, np.ndarray]]:
    strategy = getattr(cfg, "strategy", "walk_forward")
    if strategy == "walk_forward":
        return list(walk_forward_splits(n, cfg.train_size, cfg.test_size, gap=cfg.gap))
    if strategy == "rolling":
        return list(rolling_splits(n, cfg.train_size, cfg.test_size, gap=cfg.gap))
    if strategy == "expanding":
        return list(expanding_splits(n, cfg.test_size, n_splits=cfg.n_splits, gap=cfg.gap))
    if strategy == "blocked":
        return list(blocked_splits(n, n_splits=cfg.n_splits))
    if strategy == "purged_kfold":
        return list(purged_kfold(n, n_splits=cfg.n_splits, purge=cfg.purge))
    if strategy == "embargo":
        return list(embargo_splits(n, n_splits=cfg.n_splits, embargo=cfg.embargo, purge=cfg.purge))
    return list(walk_forward_splits(n, cfg.train_size, cfg.test_size, gap=cfg.gap))


def walk_forward_splits(
    n: int, train_size: int, test_size: int, *, gap: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    start = 0
    while start + train_size + gap + test_size <= n:
        tr = np.arange(start, start + train_size)
        te = np.arange(start + train_size + gap, start + train_size + gap + test_size)
        yield tr, te
        start += test_size


def rolling_splits(
    n: int, train_size: int, test_size: int, *, gap: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    yield from walk_forward_splits(n, train_size, test_size, gap=gap)


def expanding_splits(
    n: int, test_size: int, *, n_splits: int = 3, gap: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    min_train = max(n // (n_splits + 1), 10)
    for i in range(n_splits):
        te_end = n - (n_splits - i - 1) * test_size
        te_start = te_end - test_size
        tr_end = te_start - gap
        if tr_end < min_train or te_start < 0:
            continue
        yield np.arange(0, tr_end), np.arange(te_start, te_end)


def blocked_splits(n: int, *, n_splits: int = 3) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    fold = max(n // n_splits, 1)
    for i in range(n_splits):
        te = np.arange(i * fold, min((i + 1) * fold, n))
        tr = np.array([j for j in range(n) if j < te[0] or j > te[-1]], dtype=int)
        if tr.size and te.size:
            yield tr, te


def purged_kfold(
    n: int, *, n_splits: int = 3, purge: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    fold = max(n // n_splits, 1)
    for i in range(n_splits):
        te = np.arange(i * fold, min((i + 1) * fold, n))
        lo = max(te[0] - purge, 0)
        hi = min(te[-1] + purge, n - 1)
        tr = np.array([j for j in range(n) if j < lo or j > hi], dtype=int)
        if tr.size and te.size:
            yield tr, te


def embargo_splits(
    n: int, *, n_splits: int = 3, embargo: int = 5, purge: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    fold = max(n // n_splits, 1)
    for i in range(n_splits):
        te = np.arange(i * fold, min((i + 1) * fold, n))
        lo = max(te[0] - purge, 0)
        hi = min(te[-1] + purge + embargo, n - 1)
        tr = np.array([j for j in range(n) if j < lo or j > hi], dtype=int)
        if tr.size and te.size:
            yield tr, te
