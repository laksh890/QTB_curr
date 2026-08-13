"""Likelihood utilities for statistical models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]

from iqrp.app.math._array import as_array, as_vector
from iqrp.app.math.probability.distributions import Distribution
from iqrp.app.math.utils.numerical_stability import logsumexp


def log_likelihood(dist: Distribution, data: Any) -> float:
    return float(np.sum(dist.logpdf(data)))


def likelihood(dist: Distribution, data: Any) -> float:
    ll = log_likelihood(dist, data)
    return float(np.exp(ll)) if ll > -700 else 0.0


def negative_log_likelihood(dist: Distribution, data: Any) -> float:
    return float(-log_likelihood(dist, data))


def joint_log_likelihood(dists: list[Distribution], datasets: list[Any]) -> float:
    if len(dists) != len(datasets):
        raise ValueError("dists and datasets length mismatch")
    return float(sum(log_likelihood(d, x) for d, x in zip(dists, datasets, strict=True)))


def conditional_log_likelihood(
    log_joint: Any,
    log_marginal: Any,
) -> np.ndarray:
    """log p(y|x) = log p(x,y) - log p(x)."""
    return np.asarray(as_array(log_joint) - as_array(log_marginal), dtype=np.float64)


def maximum_likelihood(
    nll_fn: Callable[[np.ndarray], float],
    x0: Any,
    *,
    bounds: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Minimize negative log-likelihood via L-BFGS-B / Nelder-Mead."""
    x0_arr = as_vector(x0)
    method = "L-BFGS-B" if bounds is not None else "Nelder-Mead"
    result = optimize.minimize(nll_fn, x0_arr, method=method, bounds=bounds)
    return {
        "params": np.asarray(result.x, dtype=np.float64),
        "nll": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", 0)),
    }


def gaussian_mle(data: Any) -> tuple[float, float]:
    """Closed-form Gaussian MLE (mu, sigma)."""
    x = as_vector(data)
    mu = float(np.mean(x))
    sigma = float(np.sqrt(np.mean((x - mu) ** 2)))
    return mu, max(sigma, 1e-15)


def average_log_likelihood(dist: Distribution, data: Any) -> float:
    x = as_vector(data)
    if x.size == 0:
        return float("nan")
    return float(np.mean(dist.logpdf(x)))


def bic(nll: float, n_params: int, n_samples: int) -> float:
    return float(2.0 * nll + n_params * np.log(max(n_samples, 1)))


def aic(nll: float, n_params: int) -> float:
    return float(2.0 * nll + 2.0 * n_params)


def log_likelihood_ratio(ll_alt: float, ll_null: float) -> float:
    return float(2.0 * (ll_alt - ll_null))


def mixture_log_likelihood(component_logpdfs: Any, weights: Any) -> float:
    """Sum_t log sum_k w_k p_k(x_t). component_logpdfs shape (K, T)."""
    logs = as_array(component_logpdfs)
    w = as_vector(weights)
    w = w / w.sum()
    log_w = np.log(np.clip(w, 1e-300, None))[:, None]
    return float(np.sum(logsumexp(logs + log_w, axis=0)))
