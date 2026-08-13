"""Gaussian component densities for mixtures."""

from __future__ import annotations

import numpy as np

from iqrp.app.regimes.gmm.covariance import CovarianceType, expand_covariance

try:
    from numba import njit as _njit  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    _njit = None
else:
    _HAS_NUMBA = True


def log_gaussian_pdf(
    x: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    *,
    covariance_type: CovarianceType = "full",
) -> np.ndarray:
    """Return log N(x | mu_k, Sigma_k) with shape ``(N, K)``."""
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    mu = np.asarray(means, dtype=np.float64)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    n, d = y.shape
    k = mu.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    if covariance_type == "diag":
        var = np.clip(np.asarray(covars, dtype=np.float64), 1e-12, None)
        for j in range(k):
            diff = y - mu[j]
            out[:, j] = -0.5 * (
                d * np.log(2 * np.pi) + np.sum(np.log(var[j])) + np.sum(diff**2 / var[j], axis=1)
            )
        return out
    if covariance_type == "spherical":
        var = np.clip(np.asarray(covars, dtype=np.float64).reshape(-1), 1e-12, None)
        for j in range(k):
            diff = y - mu[j]
            out[:, j] = -0.5 * (
                d * np.log(2 * np.pi) + d * np.log(var[j]) + np.sum(diff**2, axis=1) / var[j]
            )
        return out
    # full or tied → expand to (K, D, D)
    covs = expand_covariance(covars, k, d, covariance_type)
    for j in range(k):
        out[:, j] = _full_logpdf(y, mu[j], covs[j])
    return out


def _full_logpdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = y.shape[1]
    c = np.asarray(cov, dtype=np.float64) + 1e-9 * np.eye(d)
    try:
        sign, logdet = np.linalg.slogdet(c)
        if sign <= 0:
            raise np.linalg.LinAlgError
        inv = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        c = c + 1e-3 * np.eye(d)
        _, logdet = np.linalg.slogdet(c)
        inv = np.linalg.pinv(c)
    diff = y - mean
    quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return np.asarray(-0.5 * (d * np.log(2 * np.pi) + logdet + quad), dtype=np.float64)


def sample_gaussian(
    means: np.ndarray,
    covars: np.ndarray,
    component: int,
    *,
    covariance_type: CovarianceType = "full",
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.asarray(means[component], dtype=np.float64).reshape(-1)
    d = mu.size
    if covariance_type == "diag":
        std = np.sqrt(np.clip(covars[component], 1e-12, None))
        return rng.normal(mu, std)
    if covariance_type == "spherical":
        std = float(np.sqrt(max(float(np.asarray(covars).reshape(-1)[component]), 1e-12)))
        return rng.normal(mu, std, size=d)
    if covariance_type == "tied":
        cov = np.asarray(covars, dtype=np.float64)
    else:
        cov = np.asarray(covars[component], dtype=np.float64)
    return rng.multivariate_normal(mu, cov + 1e-9 * np.eye(d))
