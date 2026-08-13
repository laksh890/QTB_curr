"""Monte Carlo framework with variance-reduction helpers."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from iqrp.app.math._array import as_vector


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    estimate: float
    std_error: float
    samples: np.ndarray
    n: int
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "std_error": self.std_error,
            "n": self.n,
            "method": self.method,
        }


class RandomStream:
    """Seeded stream factory for reproducible parallel Monte Carlo."""

    def __init__(self, seed: int = 0) -> None:
        self._ss = np.random.SeedSequence(seed)

    def spawn(self, n: int) -> list[np.random.Generator]:
        children = self._ss.spawn(n)
        return [np.random.default_rng(s) for s in children]

    def generator(self) -> np.random.Generator:
        return np.random.default_rng(self._ss.spawn(1)[0])


def monte_carlo(
    payoff: Callable[[np.random.Generator], float],
    n: int,
    *,
    seed: int = 0,
    method: str = "crude",
) -> MonteCarloResult:
    stream = RandomStream(seed)
    rng = stream.generator()
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        samples[i] = float(payoff(rng))
    return _summarize(samples, method)


def antithetic_monte_carlo(
    payoff: Callable[[float], float],
    n: int,
    *,
    sampler: Callable[[np.random.Generator], float] | None = None,
    seed: int = 0,
) -> MonteCarloResult:
    """Antithetic variates assuming symmetric noise u and -u."""
    rng = RandomStream(seed).generator()
    sampler = sampler or (lambda r: float(r.standard_normal()))
    half = max(n // 2, 1)
    samples = np.empty(half, dtype=np.float64)
    for i in range(half):
        u = sampler(rng)
        samples[i] = 0.5 * (payoff(u) + payoff(-u))
    return _summarize(samples, "antithetic")


def parallel_monte_carlo(
    payoff: Callable[[np.random.Generator], float],
    n: int,
    *,
    seed: int = 0,
    n_workers: int = 4,
) -> MonteCarloResult:
    stream = RandomStream(seed)
    workers = max(1, n_workers)
    gens = stream.spawn(workers)
    chunk = int(np.ceil(n / workers))

    def _run(rng: np.random.Generator) -> np.ndarray:
        m = chunk
        out = np.empty(m, dtype=np.float64)
        for i in range(m):
            out[i] = float(payoff(rng))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_run, gens))
    samples = np.concatenate(parts)[:n]
    return _summarize(samples, "parallel")


def control_variate(
    samples: Any,
    control: Any,
    *,
    control_mean: float,
) -> MonteCarloResult:
    """Adjust estimator using a zero-mean (known-mean) control variate."""
    y = as_vector(samples)
    c = as_vector(control)
    cov_yc = np.cov(y, c, ddof=1)[0, 1]
    var_c = float(np.var(c, ddof=1))
    beta = float(cov_yc / var_c) if var_c > 0 else 0.0
    adjusted = y - beta * (c - control_mean)
    return _summarize(adjusted, "control_variate")


def estimate_expectation(
    rvs: Callable[[int, np.random.Generator], np.ndarray],
    functional: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    seed: int = 0,
) -> MonteCarloResult:
    rng = RandomStream(seed).generator()
    x = rvs(n, rng)
    samples = np.asarray(functional(x), dtype=np.float64).ravel()
    return _summarize(samples, "expectation")


def _summarize(samples: np.ndarray, method: str) -> MonteCarloResult:
    s = as_vector(samples)
    n = len(s)
    est = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(max(n, 1)))
    return MonteCarloResult(estimate=est, std_error=se, samples=s, n=n, method=method)
