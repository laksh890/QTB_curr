"""Maximization step for Gaussian mixtures."""

from __future__ import annotations

import numpy as np

from iqrp.app.regimes.gmm.covariance import CovarianceType, estimate_covariances


def m_step(
    x: np.ndarray,
    responsibilities: np.ndarray,
    *,
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return updated ``(weights, means, covars)``."""
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    resp = np.asarray(responsibilities, dtype=np.float64)
    n = y.shape[0]
    nk = np.clip(resp.sum(axis=0), 1e-12, None)
    weights = nk / max(n, 1)
    means = (resp.T @ y) / nk[:, None]
    covars = estimate_covariances(
        y, resp, means, covariance_type=covariance_type, reg_covar=reg_covar
    )
    return weights, means, covars


def bayesian_m_step(
    x: np.ndarray,
    responsibilities: np.ndarray,
    *,
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
    weight_concentration_prior: float = 1.0,
    mean_precision_prior: float = 1.0,
    mean_prior: np.ndarray | None = None,
    covariance_prior_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variational Bayesian M-step (Dirichlet weights + Normal shrinkage)."""
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    resp = np.asarray(responsibilities, dtype=np.float64)
    n, d = y.shape
    k = resp.shape[1]
    nk = np.clip(resp.sum(axis=0), 1e-12, None)
    alpha0 = float(weight_concentration_prior)
    alpha = alpha0 + nk
    weights = alpha / alpha.sum()

    m0 = (
        np.asarray(mean_prior, dtype=np.float64).reshape(1, -1)
        if mean_prior is not None
        else y.mean(axis=0, keepdims=True)
    )
    if m0.shape[1] != d:
        m0 = y.mean(axis=0, keepdims=True)
    kappa0 = float(mean_precision_prior)
    xbar = (resp.T @ y) / nk[:, None]
    kappa = kappa0 + nk
    means = (kappa0 * m0 + nk[:, None] * xbar) / kappa[:, None]

    # cov with prior scale
    covars = estimate_covariances(
        y, resp, means, covariance_type=covariance_type, reg_covar=reg_covar
    )
    scale = float(covariance_prior_scale)
    if covariance_type == "full":
        for j in range(k):
            covars[j] = (nk[j] * covars[j] + scale * np.eye(d)) / (nk[j] + 1.0)
    elif covariance_type == "diag":
        covars = (nk[:, None] * covars + scale) / (nk[:, None] + 1.0)
    elif covariance_type == "spherical":
        covars = (nk * covars + scale) / (nk + 1.0)
    else:
        covars = (n * covars + scale * np.eye(d)) / (n + 1.0)
    return weights, means, covars
