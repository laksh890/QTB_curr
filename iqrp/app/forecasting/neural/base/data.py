"""Sequence dataset utilities for neural forecasting."""

from __future__ import annotations

import numpy as np


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised windows.

    X_seq: (N, lookback, F)
    y_seq: (N, horizon) for univariate target series aligned to rows of X/y.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lookback = max(int(lookback), 1)
    horizon = max(int(horizon), 1)
    n = min(X.shape[0], y.shape[0])
    xs, ys = [], []
    for t in range(lookback - 1, n - horizon):
        xs.append(X[t - lookback + 1 : t + 1])
        ys.append(y[t + 1 : t + 1 + horizon])
    if not xs:
        # fallback single window pad
        pad_x = np.zeros((lookback, X.shape[1] if X.ndim == 2 else 1), dtype=np.float64)
        pad_x[-min(n, lookback) :] = X[max(0, n - lookback) : n]
        return pad_x[None, ...], y[max(0, n - horizon) : n][None, ...]
    return np.stack(xs), np.stack(ys)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    n_val = max(int(n * val_ratio), 1) if n > 5 else max(n // 5, 1)
    n_val = min(n_val, n - 1) if n > 1 else 0
    if n_val <= 0:
        return X, y, X[:0], y[:0]
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return mu.astype(np.float64), sd.astype(np.float64)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    X = np.where(np.isfinite(X), X, mu)
    return (X - mu) / sd


class NumpyBatchLoader:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n = self.X.shape[0]
        idx = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            sl = idx[start : start + self.batch_size]
            yield self.X[sl], self.y[sl]

    def __len__(self) -> int:
        return int(np.ceil(self.X.shape[0] / self.batch_size))
