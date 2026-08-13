"""Expectation step for Gaussian mixtures."""

from __future__ import annotations

import numpy as np

from iqrp.app.math.utils.numerical_stability import logsumexp, stable_softmax
from iqrp.app.regimes.gmm.covariance import CovarianceType
from iqrp.app.regimes.gmm.gaussian import log_gaussian_pdf


def e_step(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    *,
    covariance_type: CovarianceType = "full",
) -> tuple[np.ndarray, float]:
    """Return responsibilities ``(N, K)`` and average log-likelihood."""
    log_prob = log_gaussian_pdf(x, means, covars, covariance_type=covariance_type)
    log_w = np.log(np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-300, None))
    log_dens = log_prob + log_w[None, :]
    log_norm = logsumexp(log_dens, axis=1)
    resp = stable_softmax(log_dens, axis=1)
    ll = float(np.mean(log_norm))
    return resp, ll


def log_likelihood(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    *,
    covariance_type: CovarianceType = "full",
) -> float:
    _, ll = e_step(x, weights, means, covars, covariance_type=covariance_type)
    y = np.asarray(x)
    n = y.shape[0] if y.ndim > 1 else y.size
    return float(ll * max(n, 1))


def pointwise_log_density(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    *,
    covariance_type: CovarianceType = "full",
) -> np.ndarray:
    log_prob = log_gaussian_pdf(x, means, covars, covariance_type=covariance_type)
    log_w = np.log(np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-300, None))
    return np.asarray(logsumexp(log_prob + log_w[None, :], axis=1), dtype=np.float64)
