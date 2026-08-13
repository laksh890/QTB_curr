"""Volatility budgeting for capital allocation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.market.correlation import covariance_matrix
from iqrp.app.risk.market.volatility import realized_volatility


def volatility_budgets(
    names: list[str],
    *,
    vols: np.ndarray | list[float] | None = None,
    returns: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    target_volatility: float = 0.10,
    vol_floor: float = 1.0e-4,
    risk_budgets: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Inverse-volatility capital weights, optionally tilted by risk_budgets.

    Historical mean returns are never used. Volatility is estimated from
    provided vols, cov diagonal, or causal realized_volatility on returns.
    """
    n = len(names)
    if n == 0:
        return {"name": "volatility_budgets", "weights": {}, "vols": {}, "budgets": {}}

    v = _resolve_vols(names, vols=vols, returns=returns, cov=cov, vol_floor=vol_floor)
    inv = 1.0 / np.maximum(v, float(vol_floor))
    if risk_budgets:
        rb = np.asarray([float(risk_budgets.get(nm, 1.0 / n)) for nm in names], dtype=np.float64)
        rb = np.maximum(rb, 0.0)
        inv = inv * rb
    w = inv / float(np.sum(inv)) if float(np.sum(inv)) > 0 else np.full(n, 1.0 / n)

    # Leverage vs target portfolio vol when cov available
    leverage = 1.0
    port_vol = None
    if cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        if c.shape == (n, n):
            port_var = float(w @ c @ w)
            port_vol = float(np.sqrt(max(port_var, 0.0)))
            if port_vol > 1e-12:
                leverage = float(np.clip(target_volatility / port_vol, 0.0, 10.0))

    weights = {names[i]: float(w[i]) for i in range(n)}
    return {
        "name": "volatility_budgets",
        "weights": weights,
        "weight_vector": w.tolist(),
        "vols": {names[i]: float(v[i]) for i in range(n)},
        "budgets": {names[i]: float(inv[i] / np.sum(inv)) for i in range(n)},
        "target_volatility": float(target_volatility),
        "portfolio_volatility": port_vol,
        "leverage_to_target": leverage,
    }


def _resolve_vols(
    names: list[str],
    *,
    vols: np.ndarray | list[float] | None,
    returns: np.ndarray | None,
    cov: np.ndarray | None,
    vol_floor: float,
) -> np.ndarray:
    n = len(names)
    floor = float(vol_floor)
    if vols is not None:
        v = np.asarray(vols, dtype=np.float64).ravel()
        if v.size == n and np.all(np.isfinite(v)) and np.all(v > 0):
            return np.maximum(v, floor)
    if cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        if c.ndim == 2 and c.shape == (n, n):
            return np.maximum(np.sqrt(np.maximum(np.diag(c), 0.0)), floor)
    if returns is not None:
        r = np.asarray(returns, dtype=np.float64)
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        if r.ndim == 2 and r.shape[1] >= n:
            out = np.empty(n, dtype=np.float64)
            for i in range(n):
                out[i] = float(realized_volatility(r[:, i], annualize=False).value)
            return np.maximum(out, floor)
        # Fallback: cov from returns
        cm = covariance_matrix(r)
        c = np.asarray(cm["matrix"], dtype=np.float64)
        if c.shape[0] >= n:
            return np.maximum(np.sqrt(np.maximum(np.diag(c)[:n], 0.0)), floor)
    return np.full(n, max(0.01, floor), dtype=np.float64)
