"""Portfolio exposure metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "beta",
    "currency_exposure",
    "factor_exposure",
    "gross_exposure",
    "leverage",
    "long_exposure",
    "net_exposure",
    "sector_exposure",
    "short_exposure",
    "summarize_exposure",
]


def _weights(weights: Any) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 0:
        return w.reshape(1)
    return w


def gross_exposure(weights: Any) -> float | np.ndarray:
    """Sum of absolute weights (scalar or per-period)."""
    w = _weights(weights)
    if w.ndim == 1:
        return float(np.sum(np.abs(w)))
    return np.sum(np.abs(w), axis=-1)


def net_exposure(weights: Any) -> float | np.ndarray:
    """Sum of signed weights."""
    w = _weights(weights)
    if w.ndim == 1:
        return float(np.sum(w))
    return np.sum(w, axis=-1)


def long_exposure(weights: Any) -> float | np.ndarray:
    """Sum of positive weights."""
    w = _weights(weights)
    if w.ndim == 1:
        return float(np.sum(np.maximum(w, 0.0)))
    return np.sum(np.maximum(w, 0.0), axis=-1)


def short_exposure(weights: Any) -> float | np.ndarray:
    """Sum of absolute negative weights."""
    w = _weights(weights)
    if w.ndim == 1:
        return float(np.sum(np.maximum(-w, 0.0)))
    return np.sum(np.maximum(-w, 0.0), axis=-1)


def leverage(weights: Any) -> float | np.ndarray:
    """Gross exposure as leverage multiple of NAV (assumes NAV=1)."""
    return gross_exposure(weights)


def beta(
    returns: Any,
    market: Any,
) -> float:
    """OLS beta of strategy returns vs market returns."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    m = np.asarray(market, dtype=np.float64).reshape(-1)
    n = min(r.size, m.size)
    if n < 2:
        return 0.0
    r = r[:n]
    m = m[:n]
    mask = np.isfinite(r) & np.isfinite(m)
    if int(np.sum(mask)) < 2:
        return 0.0
    r = r[mask]
    m = m[mask]
    var_m = float(np.var(m, ddof=1))
    if var_m < 1e-18:
        return 0.0
    cov = float(np.cov(r, m, ddof=1)[0, 1])
    return float(cov / var_m)


def factor_exposure(
    weights: Any,
    factor_loadings: Any,
) -> np.ndarray:
    """Portfolio factor exposures: ``loadings.T @ w`` or ``w @ loadings``.

    ``factor_loadings`` shape (N, K); ``weights`` shape (N,) or (T, N).
    """
    w = _weights(weights)
    L = np.asarray(factor_loadings, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("factor_loadings must be 2-D (N, K)")
    if w.ndim == 1:
        if w.size != L.shape[0]:
            raise ValueError("weights length must match factor_loadings rows")
        return L.T @ w
    if w.shape[-1] != L.shape[0]:
        raise ValueError("weights columns must match factor_loadings rows")
    return w @ L


def sector_exposure(
    weights: Any,
    sectors: Any,
) -> dict[str, float]:
    """Aggregate absolute or signed weights by sector label."""
    return _group_exposure(weights, sectors)


def currency_exposure(
    weights: Any,
    currencies: Any,
) -> dict[str, float]:
    """Aggregate signed weights by currency label."""
    return _group_exposure(weights, currencies)


def _group_exposure(weights: Any, labels: Any) -> dict[str, float]:
    w = _weights(weights)
    if w.ndim == 2:
        w = np.mean(w, axis=0)
    labs = np.asarray(labels).reshape(-1)
    if labs.size != w.size:
        raise ValueError("labels length must match weights")
    out: dict[str, float] = {}
    for lab, wi in zip(labs.tolist(), w.tolist()):
        key = str(lab)
        out[key] = out.get(key, 0.0) + float(wi)
    return out


def summarize_exposure(
    weights: Any,
    *,
    market_returns: Any | None = None,
    strategy_returns: Any | None = None,
    factor_loadings: Any | None = None,
    sectors: Any | None = None,
    currencies: Any | None = None,
) -> dict[str, Any]:
    """Exposure summary for a weight vector or weight path."""
    w = _weights(weights)
    if w.ndim == 2:
        g = float(np.mean(gross_exposure(w)))
        n = float(np.mean(net_exposure(w)))
        lo = float(np.mean(long_exposure(w)))
        sh = float(np.mean(short_exposure(w)))
    else:
        g = float(gross_exposure(w))
        n = float(net_exposure(w))
        lo = float(long_exposure(w))
        sh = float(short_exposure(w))
    out: dict[str, Any] = {
        "gross": g,
        "net": n,
        "long": lo,
        "short": sh,
        "leverage": g,
    }
    if market_returns is not None and strategy_returns is not None:
        out["beta"] = beta(strategy_returns, market_returns)
    if factor_loadings is not None:
        fe = factor_exposure(w if w.ndim == 1 else np.mean(w, axis=0), factor_loadings)
        out["factor_exposure"] = fe.tolist() if hasattr(fe, "tolist") else fe
    if sectors is not None:
        out["sector_exposure"] = sector_exposure(w, sectors)
    if currencies is not None:
        out["currency_exposure"] = currency_exposure(w, currencies)
    return out
