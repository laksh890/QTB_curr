"""Generic random-process primitives (non-financial)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_vector


def white_noise(
    n: int,
    *,
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return scale * rng.standard_normal(n)


def random_walk(
    n: int,
    *,
    x0: float = 0.0,
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    innov = white_noise(n, scale=scale, rng=rng)
    return np.concatenate([[x0], x0 + np.cumsum(innov)])


def gaussian_process_sample(
    mean: Any,
    cov: Any,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    mu = as_vector(mean)
    c = np.asarray(cov, dtype=np.float64)
    # Jitter for PSD
    c = c + 1e-10 * np.eye(c.shape[0])
    return rng.multivariate_normal(mu, c)


def ar1(
    n: int,
    *,
    phi: float = 0.8,
    sigma: float = 1.0,
    x0: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = np.empty(n, dtype=np.float64)
    out[0] = x0
    for t in range(1, n):
        out[t] = phi * out[t - 1] + sigma * rng.standard_normal()
    return out


def correlate_streams(streams: Any, correlation: Any) -> np.ndarray:
    """Apply Cholesky correlation to independent streams shape (T, K)."""
    z = np.asarray(streams, dtype=np.float64)
    corr = np.asarray(correlation, dtype=np.float64)
    chol = np.linalg.cholesky(corr)
    return np.asarray(z @ chol.T, dtype=np.float64)
