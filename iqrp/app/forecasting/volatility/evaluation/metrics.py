"""Volatility forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    rv = np.clip(np.asarray(realized_var, dtype=np.float64).reshape(-1), 1e-12, None)
    fv = np.clip(np.asarray(forecast_var, dtype=np.float64).reshape(-1), 1e-12, None)
    n = min(rv.size, fv.size)
    rv, fv = rv[:n], fv[:n]
    return float(np.mean(np.log(fv) + rv / fv))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    return float(np.sqrt(np.mean((x[:n] - y[:n]) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    return float(np.mean(np.abs(x[:n] - y[:n])))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    return float(np.mean((x[:n] - y[:n]) ** 2))


def evaluate_volatility(
    returns: np.ndarray,
    forecast_variance: np.ndarray,
    *,
    realized: np.ndarray | None = None,
) -> dict[str, float]:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    fv = np.asarray(forecast_variance, dtype=np.float64).reshape(-1)
    rv = np.asarray(realized, dtype=np.float64).reshape(-1) if realized is not None else r**2
    n = min(r.size, fv.size, rv.size)
    r, fv, rv = r[:n], fv[:n], rv[:n]
    # gaussian loglik of returns under forecast variance
    ll = float(
        -0.5
        * np.sum(np.log(2 * np.pi * np.clip(fv, 1e-12, None)) + r**2 / np.clip(fv, 1e-12, None))
    )
    return {
        "qlike": qlike(rv, fv),
        "rmse": rmse(rv, fv),
        "mae": mae(rv, fv),
        "mse": mse(rv, fv),
        "loglik": ll,
        "vol_rmse": rmse(np.sqrt(rv), np.sqrt(np.clip(fv, 1e-12, None))),
        "n": float(n),
    }


def realized_volatility(
    returns: np.ndarray, *, window: int = 21, annualization: float = 252.0
) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    w = max(int(window), 1)
    out = np.empty(r.size)
    c2 = np.cumsum(r**2)
    for t in range(r.size):
        lo = max(0, t - w + 1)
        s = c2[t] - (c2[lo - 1] if lo > 0 else 0.0)
        out[t] = np.sqrt(s / max(t - lo + 1, 1) * annualization)
    return out
