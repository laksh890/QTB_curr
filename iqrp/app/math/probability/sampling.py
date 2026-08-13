"""Sampling algorithms for the probability engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from iqrp.app.math._array import as_array, as_vector
from iqrp.app.math.utils.numerical_stability import stable_softmax


def random_sample(
    population: Any,
    size: int,
    *,
    replace: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    pop = as_array(population)
    idx = rng.choice(len(pop), size=size, replace=replace)
    return pop[idx]


def weighted_sample(
    population: Any,
    weights: Any,
    size: int,
    *,
    replace: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    pop = as_array(population)
    w = as_vector(weights).astype(np.float64)
    w = w / w.sum()
    idx = rng.choice(len(pop), size=size, replace=replace, p=w)
    return pop[idx]


def importance_sample(
    target_logpdf: Callable[[np.ndarray], np.ndarray],
    proposal_rvs: Callable[[int, np.random.Generator], np.ndarray],
    proposal_logpdf: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Self-normalized importance sampling."""
    rng = rng or np.random.default_rng()
    samples = proposal_rvs(n, rng)
    log_w = target_logpdf(samples) - proposal_logpdf(samples)
    weights = stable_softmax(log_w)
    return {"samples": samples, "weights": weights, "log_weights": log_w}


def stratified_sample(
    n: int,
    *,
    low: float = 0.0,
    high: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Stratified uniforms on [low, high]."""
    rng = rng or np.random.default_rng()
    strata = (np.arange(n) + rng.random(n)) / n
    return low + (high - low) * strata


def bootstrap_sample(
    data: Any,
    *,
    n_bootstrap: int = 1000,
    statistic: Callable[[np.ndarray], float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    x = as_vector(data)
    n = len(x)
    if statistic is None:
        out = np.empty((n_bootstrap, n), dtype=np.float64)
        for i in range(n_bootstrap):
            out[i] = x[rng.integers(0, n, size=n)]
        return out
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        stats[i] = float(statistic(x[rng.integers(0, n, size=n)]))
    return stats


def monte_carlo_sample(
    rvs: Callable[[int, np.random.Generator], np.ndarray],
    n: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rvs(n, rng)


def rejection_sample(
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_rvs: Callable[[int, np.random.Generator], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    m: float,
    n: int,
    *,
    rng: np.random.Generator | None = None,
    max_trials: int | None = None,
) -> np.ndarray:
    """Rejection sampling with envelope constant ``m``."""
    rng = rng or np.random.default_rng()
    accepted: list[float] = []
    trials = 0
    limit = max_trials if max_trials is not None else max(n * 100, 1000)
    while len(accepted) < n and trials < limit:
        batch = min(n - len(accepted), 256)
        cand = proposal_rvs(batch, rng).ravel()
        u = rng.random(batch)
        accept = u * m * proposal_pdf(cand) <= target_pdf(cand)
        accepted.extend(cand[accept].tolist())
        trials += batch
    return np.asarray(accepted[:n], dtype=np.float64)


def systematic_resample(weights: Any, rng: np.random.Generator | None = None) -> np.ndarray:
    """Systematic resampling indices for particle filters."""
    rng = rng or np.random.default_rng()
    w = as_vector(weights).astype(np.float64)
    w = w / w.sum()
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(w)
    return np.searchsorted(cumsum, positions).astype(np.int64)
