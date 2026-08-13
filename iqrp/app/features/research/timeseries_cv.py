"""Time-series validation splits — never shuffle."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from iqrp.app.features.research.config import PredictiveConfig


@dataclass(frozen=True, slots=True)
class TimeSeriesSplit:
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


def iter_splits(n_rows: int, cfg: PredictiveConfig) -> Iterator[TimeSeriesSplit]:
    mode = cfg.evaluation_mode
    if mode == "blocked":
        yield from _blocked(n_rows, cfg)
    elif mode == "expanding":
        yield from _expanding(n_rows, cfg)
    elif mode == "rolling":
        yield from _rolling(n_rows, cfg)
    else:
        yield from _walk_forward(n_rows, cfg)


def _walk_forward(n_rows: int, cfg: PredictiveConfig) -> Iterator[TimeSeriesSplit]:
    train_end = cfg.min_train_size
    while train_end + cfg.test_size <= n_rows:
        yield TimeSeriesSplit(0, train_end, train_end, train_end + cfg.test_size)
        train_end += cfg.step_size


def _expanding(n_rows: int, cfg: PredictiveConfig) -> Iterator[TimeSeriesSplit]:
    # Alias of walk-forward with origin fixed at 0 (already the case).
    yield from _walk_forward(n_rows, cfg)


def _rolling(n_rows: int, cfg: PredictiveConfig) -> Iterator[TimeSeriesSplit]:
    train_end = cfg.min_train_size
    while train_end + cfg.test_size <= n_rows:
        train_start = max(0, train_end - cfg.min_train_size)
        yield TimeSeriesSplit(train_start, train_end, train_end, train_end + cfg.test_size)
        train_end += cfg.step_size


def _blocked(n_rows: int, cfg: PredictiveConfig) -> Iterator[TimeSeriesSplit]:
    n_splits = max(2, cfg.blocked_n_splits)
    block = n_rows // n_splits
    purge = max(0, cfg.blocked_purge)
    if block <= purge + 2:
        return
    for i in range(n_splits):
        test_start = i * block
        test_end = n_rows if i == n_splits - 1 else (i + 1) * block
        # Train = all other blocks excluding purge gap around test.
        train_idx_end_left = max(0, test_start - purge)
        train_idx_start_right = min(n_rows, test_end + purge)
        # Emit as contiguous train before test when possible; otherwise skip tiny.
        if train_idx_end_left >= cfg.min_train_size:
            yield TimeSeriesSplit(0, train_idx_end_left, test_start, test_end)
        elif n_rows - train_idx_start_right >= cfg.min_train_size:
            # Use post-test block as train and earlier as test (still ordered eval
            # on the held-out segment without shuffling).
            yield TimeSeriesSplit(train_idx_start_right, n_rows, test_start, test_end)
