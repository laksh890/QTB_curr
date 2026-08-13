"""Accuracy / speed / memory benchmarks for the math engine."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math.matrices import cholesky, eig, multiply, svd
from iqrp.app.math.probability import gaussian, log_likelihood
from iqrp.app.math.statistics import mean, pearson, variance
from iqrp.app.math.utils import logsumexp, stable_softmax


@dataclass(frozen=True, slots=True)
class BenchResult:
    name: str
    seconds: float
    peak_kib: float
    metric: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seconds": self.seconds,
            "peak_kib": self.peak_kib,
            "metric": self.metric,
            "details": dict(self.details),
        }


def _timed(fn: Callable[[], Any]) -> tuple[Any, float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, elapsed, peak / 1024.0


def accuracy_benchmarks(rng: np.random.Generator | None = None) -> list[BenchResult]:
    rng = rng or np.random.default_rng(0)
    x = rng.normal(size=5000)
    results: list[BenchResult] = []

    def mean_err() -> float:
        return abs(float(mean(x)) - float(np.mean(x)))

    err, sec, mem = _timed(mean_err)
    results.append(BenchResult("mean_vs_numpy", sec, mem, err, {"tol": 1e-12}))

    def var_err() -> float:
        return abs(float(variance(x, ddof=1)) - float(np.var(x, ddof=1)))

    err, sec, mem = _timed(var_err)
    results.append(BenchResult("variance_vs_numpy", sec, mem, err, {"tol": 1e-10}))

    y = rng.normal(size=5000)

    def corr_err() -> float:
        return abs(pearson(x, y) - float(stats.pearsonr(x, y).statistic))

    err, sec, mem = _timed(corr_err)
    results.append(BenchResult("pearson_vs_scipy", sec, mem, err, {"tol": 1e-10}))

    dist = gaussian(0.0, 1.0)

    def ll_err() -> float:
        ours = log_likelihood(dist, x[:200])
        theirs = float(np.sum(stats.norm.logpdf(x[:200])))
        return abs(ours - theirs)

    err, sec, mem = _timed(ll_err)
    results.append(BenchResult("gaussian_ll_vs_scipy", sec, mem, err, {"tol": 1e-8}))
    return results


def speed_benchmarks(rng: np.random.Generator | None = None) -> list[BenchResult]:
    rng = rng or np.random.default_rng(1)
    a = rng.normal(size=(400, 400))
    a = a @ a.T + np.eye(400)
    results: list[BenchResult] = []

    _, sec, mem = _timed(lambda: multiply(a, a))
    results.append(BenchResult("matmul_400", sec, mem, sec, {}))
    _, sec, mem = _timed(lambda: cholesky(a))
    results.append(BenchResult("cholesky_400", sec, mem, sec, {}))
    _, sec, mem = _timed(lambda: eig(a))
    results.append(BenchResult("eig_400", sec, mem, sec, {}))
    _, sec, mem = _timed(lambda: svd(a[:, :50]))
    results.append(BenchResult("svd_400x50", sec, mem, sec, {}))
    z = rng.normal(size=(2000, 32))
    _, sec, mem = _timed(lambda: stable_softmax(z, axis=1))
    results.append(BenchResult("softmax_2000x32", sec, mem, sec, {}))
    _, sec, mem = _timed(lambda: logsumexp(z, axis=1))
    results.append(BenchResult("logsumexp_2000x32", sec, mem, sec, {}))
    return results


def memory_benchmarks(rng: np.random.Generator | None = None) -> list[BenchResult]:
    rng = rng or np.random.default_rng(2)
    results: list[BenchResult] = []

    def big_cov() -> float:
        x = rng.normal(size=(5000, 20))
        return float(np.linalg.norm(np.cov(x, rowvar=False)))

    val, sec, mem = _timed(big_cov)
    results.append(BenchResult("cov_5000x20", sec, mem, val, {}))

    def mc() -> float:
        from iqrp.app.math.stochastic import monte_carlo

        r = monte_carlo(lambda g: float(g.standard_normal() ** 2), 20_000, seed=3)
        return r.estimate

    val, sec, mem = _timed(mc)
    results.append(BenchResult("monte_carlo_20k", sec, mem, val, {"expected": 1.0}))
    return results


def run_all_benchmarks() -> dict[str, list[dict[str, Any]]]:
    return {
        "accuracy": [r.to_dict() for r in accuracy_benchmarks()],
        "speed": [r.to_dict() for r in speed_benchmarks()],
        "memory": [r.to_dict() for r in memory_benchmarks()],
    }
