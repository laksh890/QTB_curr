"""Synthetic statistical processes for recovery / simulation validation."""

from __future__ import annotations

from typing import Any

import numpy as np


def simulate_ar(
    n: int,
    ar: list[float] | np.ndarray,
    *,
    sigma: float = 1.0,
    intercept: float = 0.0,
    burn: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    gen = rng or np.random.default_rng()
    phi = np.asarray(ar, dtype=np.float64).reshape(-1)
    p = phi.size
    total = n + burn
    e = gen.normal(0, sigma, size=total)
    y = np.zeros(total, dtype=np.float64)
    for t in range(total):
        y[t] = intercept + e[t]
        for i in range(min(p, t)):
            y[t] += phi[i] * y[t - 1 - i]
    return y[burn:]


def simulate_ma(
    n: int,
    ma: list[float] | np.ndarray,
    *,
    sigma: float = 1.0,
    intercept: float = 0.0,
    burn: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    gen = rng or np.random.default_rng()
    theta = np.asarray(ma, dtype=np.float64).reshape(-1)
    q = theta.size
    total = n + burn
    e = gen.normal(0, sigma, size=total)
    y = np.zeros(total, dtype=np.float64)
    for t in range(total):
        y[t] = intercept + e[t]
        for j in range(min(q, t)):
            y[t] += theta[j] * e[t - 1 - j]
    return y[burn:]


def simulate_arma(
    n: int,
    ar: list[float] | np.ndarray,
    ma: list[float] | np.ndarray,
    *,
    sigma: float = 1.0,
    intercept: float = 0.0,
    burn: int = 80,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    gen = rng or np.random.default_rng()
    phi = np.asarray(ar, dtype=np.float64).reshape(-1)
    theta = np.asarray(ma, dtype=np.float64).reshape(-1)
    p, q = phi.size, theta.size
    total = n + burn
    e = gen.normal(0, sigma, size=total)
    y = np.zeros(total, dtype=np.float64)
    for t in range(total):
        y[t] = intercept + e[t]
        for i in range(min(p, t)):
            y[t] += phi[i] * y[t - 1 - i]
        for j in range(min(q, t)):
            y[t] += theta[j] * e[t - 1 - j]
    return y[burn:]


def simulate_arima(
    n: int,
    ar: list[float] | np.ndarray,
    d: int,
    ma: list[float] | np.ndarray,
    *,
    sigma: float = 1.0,
    intercept: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    z = simulate_arma(n + max(int(d), 0) * 5, ar, ma, sigma=sigma, intercept=intercept, rng=rng)
    y = z
    for _ in range(max(int(d), 0)):
        y = np.cumsum(y)
    return y[-n:]


def simulate_seasonal_arima(
    n: int,
    *,
    period: int = 12,
    ar: list[float] | np.ndarray = (0.5,),
    ma: list[float] | np.ndarray = (),
    seasonal_ar: list[float] | np.ndarray = (0.4,),
    d: int = 0,
    D: int = 1,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a simple seasonal process via SAR filter approximation."""
    gen = rng or np.random.default_rng()
    s = max(int(period), 2)
    total = n + 5 * s
    e = gen.normal(0, sigma, size=total)
    y = np.zeros(total, dtype=np.float64)
    phi = np.asarray(ar, dtype=np.float64).reshape(-1)
    Phi = np.asarray(seasonal_ar, dtype=np.float64).reshape(-1)
    theta = np.asarray(ma, dtype=np.float64).reshape(-1)
    for t in range(total):
        y[t] = e[t]
        for i in range(min(phi.size, t)):
            y[t] += phi[i] * y[t - 1 - i]
        for i in range(min(Phi.size, t // s)):
            y[t] += Phi[i] * y[t - s * (i + 1)]
        for j in range(min(theta.size, t)):
            y[t] += theta[j] * e[t - 1 - j]
    # seasonal integration
    for _ in range(max(int(D), 0)):
        out = np.zeros_like(y)
        out[:s] = y[:s]
        for t in range(s, y.size):
            out[t] = out[t - s] + y[t]
        y = out
    for _ in range(max(int(d), 0)):
        y = np.cumsum(y)
    return y[-n:]


def simulate_var(
    n: int,
    coefs: np.ndarray,
    *,
    intercept: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    burn: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate VAR(p). ``coefs`` shape (p, K, K)."""
    gen = rng or np.random.default_rng()
    p, K, _ = coefs.shape
    c = np.zeros(K) if intercept is None else np.asarray(intercept, dtype=np.float64)
    if sigma is None:
        sigma = np.eye(K)
    total = n + burn
    y = np.zeros((total, K), dtype=np.float64)
    for t in range(total):
        eps = gen.multivariate_normal(np.zeros(K), sigma)
        y[t] = c + eps
        for lag in range(min(p, t)):
            y[t] = y[t] + coefs[lag] @ y[t - 1 - lag]
    return y[burn:]


def simulate_cointegrated_pair(
    n: int,
    *,
    beta: float = 1.0,
    phi: float = 0.8,
    sigma: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Two I(1) series with cointegrating relation y - beta x ~ AR(1)."""
    gen = rng or np.random.default_rng()
    x = np.cumsum(gen.normal(0, sigma, size=n))
    spread = simulate_ar(n, [phi], sigma=sigma * 0.5, rng=gen)
    y = beta * x + spread
    return np.column_stack([y, x])


def to_frame(
    y: np.ndarray,
    *,
    target: str = "target",
    prefix: str = "y",
    regime: np.ndarray | None = None,
) -> Any:
    import polars as pl

    arr = np.asarray(y, dtype=np.float64)
    n = arr.shape[0]
    data: dict[str, Any] = {"open_time": list(range(n))}
    if arr.ndim == 1:
        data[target] = arr
        data["f0"] = arr
    else:
        for j in range(arr.shape[1]):
            data[f"{prefix}{j}"] = arr[:, j]
        data[target] = arr[:, 0]
    if regime is not None:
        data["regime"] = np.asarray(regime).reshape(-1)[:n]
    return pl.DataFrame(data)
