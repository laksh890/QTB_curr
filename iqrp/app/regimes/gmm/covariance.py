"""Covariance parameterizations for Gaussian mixtures."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

CovarianceType = Literal["full", "diag", "tied", "spherical"]


def n_covariance_params(n_components: int, n_features: int, cov_type: CovarianceType) -> int:
    d = int(n_features)
    k = int(n_components)
    if cov_type == "full":
        return k * d * (d + 1) // 2
    if cov_type == "diag":
        return k * d
    if cov_type == "tied":
        return d * (d + 1) // 2
    return k  # spherical


def estimate_covariances(
    x: np.ndarray,
    responsibilities: np.ndarray,
    means: np.ndarray,
    *,
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
) -> np.ndarray:
    """Weighted M-step covariance estimate for the selected parameterization."""
    y = np.asarray(x, dtype=np.float64)
    resp = np.asarray(responsibilities, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    n, d = y.shape
    k = mu.shape[0]
    nk = np.clip(resp.sum(axis=0), 1e-12, None)
    reg = float(reg_covar)

    if covariance_type == "full":
        cov = np.empty((k, d, d), dtype=np.float64)
        for j in range(k):
            diff = y - mu[j]
            weighted = diff * np.sqrt(resp[:, j])[:, None]
            c = (weighted.T @ weighted) / nk[j]
            c = 0.5 * (c + c.T) + reg * np.eye(d)
            cov[j] = c
        return cov

    if covariance_type == "diag":
        cov = np.empty((k, d), dtype=np.float64)
        for j in range(k):
            diff = y - mu[j]
            cov[j] = np.clip((resp[:, j][:, None] * diff**2).sum(axis=0) / nk[j], reg, None)
        return cov

    if covariance_type == "spherical":
        cov = np.empty(k, dtype=np.float64)
        for j in range(k):
            diff = y - mu[j]
            cov[j] = max(float((resp[:, j][:, None] * diff**2).sum() / (nk[j] * d)), reg)
        return cov

    # tied
    cov = np.zeros((d, d), dtype=np.float64)
    for j in range(k):
        diff = y - mu[j]
        weighted = diff * np.sqrt(resp[:, j])[:, None]
        cov += weighted.T @ weighted
    cov = cov / max(float(n), 1.0)
    cov = 0.5 * (cov + cov.T) + reg * np.eye(d)
    return cov


def covariance_to_dict(covars: np.ndarray, covariance_type: CovarianceType) -> Any:
    return np.asarray(covars, dtype=np.float64).tolist()


def covariance_from_dict(data: Any, covariance_type: CovarianceType) -> np.ndarray:
    return np.asarray(data, dtype=np.float64)


def component_covariance(
    covars: np.ndarray, index: int, covariance_type: CovarianceType
) -> np.ndarray:
    """Return a (d, d) covariance matrix for component ``index``."""
    if covariance_type == "full":
        return np.asarray(covars[index], dtype=np.float64)
    if covariance_type == "tied":
        return np.asarray(covars, dtype=np.float64)
    if covariance_type == "diag":
        return np.diag(np.asarray(covars[index], dtype=np.float64))
    # spherical
    d = 1
    val = float(covars) if covars.ndim == 0 else float(covars[index])
    # infer d from means context — caller may expand; return 1x1 if unknown
    return np.array([[val]], dtype=np.float64) if d == 1 else val * np.eye(d)


def expand_covariance(
    covars: np.ndarray,
    n_components: int,
    n_features: int,
    covariance_type: CovarianceType,
) -> np.ndarray:
    """Expand stored covars to shape ``(K, D, D)``."""
    k, d = int(n_components), int(n_features)
    out = np.empty((k, d, d), dtype=np.float64)
    if covariance_type == "full":
        arr = np.asarray(covars, dtype=np.float64)
        if arr.ndim == 2 and arr.shape == (d, d):
            for j in range(k):
                out[j] = arr
            return out
        return arr
    if covariance_type == "tied":
        tied = np.asarray(covars, dtype=np.float64)
        for j in range(k):
            out[j] = tied
        return out
    if covariance_type == "diag":
        for j in range(k):
            out[j] = np.diag(np.asarray(covars[j], dtype=np.float64))
        return out
    for j in range(k):
        out[j] = float(covars[j]) * np.eye(d)
    return out
