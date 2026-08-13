"""Risk-parity / equal-risk-contribution weights."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import as_weights


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto probability simplex (non-negative, sum=1)."""
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho_idx = int(rho[-1])
        theta = (cssv[rho_idx] - 1.0) / float(rho_idx + 1)
    w = np.maximum(v - theta, 0.0)
    s = float(np.sum(w))
    if s <= 0:
        return np.full(n, 1.0 / n)
    return w / s


def risk_parity_weights(
    cov: Any,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Long-only risk-parity weights via iterative ERC update."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    if n == 0:
        return {"name": "risk_parity_weights", "weights": [], "converged": True, "iterations": 0}

    # Stabilize
    c = 0.5 * (c + c.T)
    diag = np.maximum(np.diag(c), 1e-12)
    w = 1.0 / np.sqrt(diag)
    w = w / float(np.sum(w))

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        port_var = float(w @ c @ w)
        if port_var <= 1e-18:
            break
        mrc = (c @ w) / np.sqrt(port_var)
        crc = w * mrc
        target = float(np.mean(crc))
        # Multiplicative update toward equal CRC
        with np.errstate(divide="ignore", invalid="ignore"):
            adj = np.where(crc > 0, target / crc, 1.0)
        w_new = w * adj
        w_new = _project_simplex(w_new)
        if float(np.max(np.abs(w_new - w))) < tol:
            w = w_new
            converged = True
            break
        w = w_new

    return {
        "name": "risk_parity_weights",
        "weights": w.tolist(),
        "converged": converged,
        "iterations": it,
        "n_assets": n,
    }


def equal_risk_contribution(
    cov: Any,
    *,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Alias for risk_parity_weights with component risk diagnostics."""
    from iqrp.app.risk.portfolio.portfolio_risk import component_risk_contribution

    result = risk_parity_weights(cov, max_iter=max_iter, tol=tol)
    w = as_weights(result["weights"])
    crc = component_risk_contribution(w, cov)
    result = dict(result)
    result["name"] = "equal_risk_contribution"
    result["component_risk_contribution"] = crc
    return result
