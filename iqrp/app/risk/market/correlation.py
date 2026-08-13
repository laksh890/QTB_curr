"""Correlation estimators for multi-asset returns."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns


def _as_return_matrix(returns: Any) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("returns must be 1-D or 2-D (T x N)")
    return arr


def correlation_matrix(
    returns: Any,
    *,
    window: int | None = None,
) -> dict[str, Any]:
    """Pearson correlation matrix on trailing observations."""
    x = _as_return_matrix(returns)
    t, n = x.shape
    if window is not None and window > 0:
        w = min(int(window), t)
        x = x[-w:]
        t = w
    if t < 2 or n == 0:
        corr = np.eye(max(n, 1), dtype=np.float64) if n else np.zeros((0, 0))
    else:
        # Pairwise finite mask per column pair via nan-safe corrcoef on cleaned rows
        mask = np.all(np.isfinite(x), axis=1)
        clean = x[mask]
        if clean.shape[0] < 2:
            corr = np.eye(n, dtype=np.float64)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(clean, rowvar=False)
            if corr.ndim == 0:
                corr = np.array([[1.0]], dtype=np.float64)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr, 1.0)
    return {
        "name": "correlation_matrix",
        "matrix": corr.tolist(),
        "shape": list(corr.shape),
        "n_obs": int(t),
        "window": window,
        "method": "pearson",
    }


def ewma_correlation(
    returns: Any,
    *,
    lambda_: float = 0.94,
) -> dict[str, Any]:
    """EWMA correlation matrix (RiskMetrics-style), causal recursion."""
    x = _as_return_matrix(returns)
    t, n = x.shape
    lam = float(np.clip(lambda_, 1e-6, 1.0 - 1e-6))
    if t == 0 or n == 0:
        return {
            "name": "ewma_correlation",
            "matrix": [],
            "shape": [0, 0],
            "n_obs": 0,
            "lambda": lam,
            "method": "ewma",
        }

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # Initialize with outer product of first row
    cov = np.outer(x[0], x[0])
    for i in range(1, t):
        ri = x[i]
        cov = lam * cov + (1.0 - lam) * np.outer(ri, ri)

    vol = np.sqrt(np.maximum(np.diag(cov), 0.0))
    denom = np.outer(vol, vol)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    return {
        "name": "ewma_correlation",
        "matrix": corr.tolist(),
        "shape": list(corr.shape),
        "n_obs": int(t),
        "lambda": lam,
        "method": "ewma",
        "volatilities": vol.tolist(),
    }


def covariance_matrix(
    returns: Any,
    *,
    window: int | None = None,
) -> dict[str, Any]:
    """Sample covariance matrix on trailing observations."""
    x = _as_return_matrix(returns)
    t, n = x.shape
    if window is not None and window > 0:
        w = min(int(window), t)
        x = x[-w:]
        t = w
    if t < 2 or n == 0:
        cov = np.zeros((max(n, 0), max(n, 0)), dtype=np.float64)
    else:
        mask = np.all(np.isfinite(x), axis=1)
        clean = x[mask]
        if clean.shape[0] < 2:
            cov = np.zeros((n, n), dtype=np.float64)
        else:
            cov = np.cov(clean, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]], dtype=np.float64)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "name": "covariance_matrix",
        "matrix": cov.tolist(),
        "shape": list(cov.shape),
        "n_obs": int(t),
        "window": window,
        "method": "sample",
    }


def shrinkage_covariance(
    returns: Any,
    *,
    intensity: float | None = None,
) -> dict[str, Any]:
    """Ledoit–Wolf-style shrinkage toward scaled identity (causal sample only)."""
    sample = covariance_matrix(returns)
    cov = np.asarray(sample["matrix"], dtype=np.float64)
    n = cov.shape[0]
    if n == 0:
        return {**sample, "name": "shrinkage_covariance", "method": "shrinkage", "intensity": 0.0}
    x = _as_return_matrix(returns)
    t = x.shape[0]
    mu = float(np.trace(cov) / n) if n else 0.0
    target = mu * np.eye(n)
    if intensity is None:
        # Simple intensity heuristic in [0, 1]
        fro = float(np.linalg.norm(cov - target, "fro"))
        intensity = float(np.clip(fro / (fro + mu * n + 1e-12), 0.0, 1.0))
    alpha = float(np.clip(intensity, 0.0, 1.0))
    shrunk = (1.0 - alpha) * cov + alpha * target
    return {
        "name": "shrinkage_covariance",
        "matrix": shrunk.tolist(),
        "shape": list(shrunk.shape),
        "n_obs": int(t),
        "intensity": alpha,
        "method": "shrinkage_identity",
        "sample_method": sample["method"],
    }


def ewma_covariance(
    returns: Any,
    *,
    lambda_: float = 0.94,
) -> dict[str, Any]:
    """EWMA covariance matrix (RiskMetrics-style), causal recursion."""
    x = _as_return_matrix(returns)
    t, n = x.shape
    lam = float(np.clip(lambda_, 1e-6, 1.0 - 1e-6))
    if t == 0 or n == 0:
        return {
            "name": "ewma_covariance",
            "matrix": [],
            "shape": [0, 0],
            "n_obs": 0,
            "lambda": lam,
            "method": "ewma",
        }
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.outer(x[0], x[0])
    for i in range(1, t):
        ri = x[i]
        cov = lam * cov + (1.0 - lam) * np.outer(ri, ri)
    return {
        "name": "ewma_covariance",
        "matrix": cov.tolist(),
        "shape": list(cov.shape),
        "n_obs": int(t),
        "lambda": lam,
        "method": "ewma",
    }


def rolling_correlation(
    x: Any,
    y: Any,
    *,
    window: int = 60,
) -> RiskMeasure:
    """Latest rolling Pearson correlation between two series."""
    a = as_returns(x)
    b = as_returns(y)
    n = int(min(a.size, b.size))
    w = max(int(window), 2)
    if n < w:
        value = 0.0
        used = n
    else:
        a = a[-w:]
        b = b[-w:]
        used = w
        if np.std(a) <= 0 or np.std(b) <= 0:
            value = 0.0
        else:
            value = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(value):
                value = 0.0

    return RiskMeasure(
        name="rolling_correlation",
        value=value,
        unit="correlation",
        method="pearson_rolling",
        parameters={"window": w, "n_obs": used},
    )
