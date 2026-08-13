"""Evaluate capital allocations on diversification / budget match / capacity — not alpha."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.portfolio.diversification import diversification_ratio


def evaluate_allocation(
    weights: np.ndarray | list[float] | dict[str, float],
    *,
    names: list[str] | None = None,
    cov: np.ndarray | None = None,
    risk_budgets: dict[str, float] | np.ndarray | list[float] | None = None,
    capacity_scales: dict[str, float] | None = None,
    capital: float = 1.0,
    max_notional: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score allocation quality without using alpha / historical mean returns.

    Metrics
    -------
    diversification :
        Diversification ratio when cov provided.
    risk_budget_error :
        L2 error between realized risk contributions and target budgets.
    capacity_utilization :
        Mean notional / max_notional (or inverse of capacity scales proxy).
    """
    if isinstance(weights, dict):
        keys = names or list(weights.keys())
        w = np.asarray([float(weights.get(k, 0.0)) for k in keys], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        keys = names if names and len(names) == w.size else [f"s{i}" for i in range(w.size)]
    n = w.size
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s > 0:
        w = w / s

    # Diversification
    div_value = None
    if cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        if c.shape == (n, n):
            div_value = float(diversification_ratio(w, c).value)

    # Risk budget matching via marginal risk contributions when cov available
    budget_error = None
    realized_rc: dict[str, float] = {}
    target_rb: dict[str, float] = {}
    if risk_budgets is not None:
        if isinstance(risk_budgets, dict):
            tb = np.asarray([float(risk_budgets.get(k, 0.0)) for k in keys], dtype=np.float64)
        else:
            tb = np.asarray(risk_budgets, dtype=np.float64).ravel()
            if tb.size != n:
                tb = np.full(n, 1.0 / n)
        tb = np.maximum(tb, 0.0)
        if float(np.sum(tb)) > 0:
            tb = tb / float(np.sum(tb))
        target_rb = {keys[i]: float(tb[i]) for i in range(n)}

        if cov is not None:
            c = np.asarray(cov, dtype=np.float64)
            if c.shape == (n, n):
                port_var = float(w @ c @ w)
                if port_var > 1e-18:
                    mrc = c @ w
                    crc = w * mrc
                    rc = crc / float(np.sum(crc)) if float(np.sum(crc)) > 0 else tb
                else:
                    rc = w.copy()
                realized_rc = {keys[i]: float(rc[i]) for i in range(n)}
                budget_error = float(np.sqrt(np.mean((rc - tb) ** 2)))
            else:
                budget_error = float(np.sqrt(np.mean((w - tb) ** 2)))
                realized_rc = {keys[i]: float(w[i]) for i in range(n)}
        else:
            budget_error = float(np.sqrt(np.mean((w - tb) ** 2)))
            realized_rc = {keys[i]: float(w[i]) for i in range(n)}

    # Capacity utilization
    util = None
    if max_notional:
        notionals = {keys[i]: float(capital) * float(w[i]) for i in range(n)}
        ratios = []
        for k in keys:
            mx = float(max_notional.get(k, 0.0))
            if mx > 1e-12:
                ratios.append(notionals[k] / mx)
        util = float(np.mean(ratios)) if ratios else None
    elif capacity_scales:
        # High scale → spare capacity; utilization proxy = 1 - mean(scale)
        vals = [float(capacity_scales.get(k, 1.0)) for k in keys]
        util = float(1.0 - np.mean(vals))

    # Composite score in [0, 1]: higher better (NOT alpha)
    pieces: list[float] = []
    if div_value is not None:
        # Diversification ratio typically in [1, sqrt(n)]; map softly
        pieces.append(float(np.clip((div_value - 1.0) / max(np.sqrt(n) - 1.0, 1e-6), 0.0, 1.0)))
    if budget_error is not None:
        pieces.append(float(np.clip(1.0 - budget_error * 5.0, 0.0, 1.0)))
    if util is not None:
        # Prefer utilization in [0.2, 0.8]
        pieces.append(float(np.clip(1.0 - abs(util - 0.5) * 2.0, 0.0, 1.0)))
    score = float(np.mean(pieces)) if pieces else 0.0

    return {
        "name": "evaluate_allocation",
        "score": score,
        "diversification_ratio": div_value,
        "risk_budget_error": budget_error,
        "capacity_utilization": util,
        "realized_risk_contribution": realized_rc,
        "target_risk_budgets": target_rb,
        "notes": "Evaluation excludes alpha / historical mean return metrics",
    }
