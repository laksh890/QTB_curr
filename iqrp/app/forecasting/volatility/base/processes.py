"""Synthetic stochastic volatility processes for recovery validation."""

from __future__ import annotations

from typing import Any

import numpy as np


def simulate_garch(
    n: int,
    *,
    omega: float = 0.05,
    alpha: float = 0.1,
    beta: float = 0.85,
    burn: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gen = rng or np.random.default_rng()
    total = n + burn
    eps = np.zeros(total)
    var = np.zeros(total)
    var[0] = omega / max(1 - alpha - beta, 1e-6)
    for t in range(1, total):
        var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(max(var[t], 1e-12)) * gen.normal()
    return eps[burn:], var[burn:]


def simulate_gjr(
    n: int,
    *,
    omega: float = 0.05,
    alpha: float = 0.05,
    gamma: float = 0.1,
    beta: float = 0.85,
    burn: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gen = rng or np.random.default_rng()
    total = n + burn
    eps = np.zeros(total)
    var = np.zeros(total)
    var[0] = omega / max(1 - alpha - 0.5 * gamma - beta, 1e-6)
    for t in range(1, total):
        ind = 1.0 if eps[t - 1] < 0 else 0.0
        var[t] = omega + (alpha + gamma * ind) * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = np.sqrt(max(var[t], 1e-12)) * gen.normal()
    return eps[burn:], var[burn:]


def simulate_dcc(
    n: int,
    *,
    k: int = 2,
    burn: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (T,K) and true correlation path for asset 0,1."""
    gen = rng or np.random.default_rng()
    total = n + burn
    rets = np.zeros((total, k))
    corr = np.zeros(total)
    q11 = q22 = 1.0
    q12 = 0.3
    a, b = 0.05, 0.9
    for t in range(total):
        scale = np.sqrt(q11 * q22)
        rho = q12 / max(scale, 1e-12)
        rho = float(np.clip(rho, -0.99, 0.99))
        corr[t] = rho
        cov = np.array([[1.0, rho], [rho, 1.0]])
        if k > 2:
            mat = np.eye(k)
            mat[:2, :2] = cov
            cov = mat
        z = gen.multivariate_normal(np.zeros(k), cov)
        # constant vol
        rets[t] = z * 0.02
        # DCC update on standardized residuals ≈ z
        q11 = (1 - a - b) * 1.0 + a * z[0] ** 2 + b * q11
        q22 = (1 - a - b) * 1.0 + a * z[1] ** 2 + b * q22
        q12 = (1 - a - b) * 0.3 + a * z[0] * z[1] + b * q12
    return rets[burn:], corr[burn:]


def to_returns_frame(
    returns: np.ndarray,
    *,
    target: str = "returns",
    prefix: str = "r",
    regime: np.ndarray | None = None,
) -> Any:
    import polars as pl

    arr = np.asarray(returns, dtype=np.float64)
    n = arr.shape[0]
    data: dict[str, Any] = {"open_time": list(range(n))}
    if arr.ndim == 1:
        data[target] = arr
    else:
        for j in range(arr.shape[1]):
            data[f"{prefix}{j}"] = arr[:, j]
        data[target] = arr[:, 0]
    if regime is not None:
        data["regime"] = np.asarray(regime).reshape(-1)[:n]
    return pl.DataFrame(data)
