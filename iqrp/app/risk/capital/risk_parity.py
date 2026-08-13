"""Capital-level risk parity wrapping ``iqrp.app.risk.sizing.risk_parity``."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.sizing.risk_parity import risk_parity_weights


def capital_risk_parity(
    cov: Any,
    *,
    names: list[str] | None = None,
    risk_budgets: dict[str, float] | np.ndarray | list[float] | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Risk-parity / budgeted risk-parity weights for capital allocation.

    Delegates the ERC core to ``risk_parity_weights``. When ``risk_budgets`` are
    provided, applies budget-proportional scaling on top of the ERC solution
    (capital-level budgeting), then renormalizes.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    keys = names if names and len(names) == n else [f"s{i}" for i in range(n)]

    base = risk_parity_weights(c, max_iter=max_iter, tol=tol)
    w = np.asarray(base.get("weights") or [], dtype=np.float64)
    if w.size != n:
        w = np.full(n, 1.0 / n if n else 0.0)

    budget_applied = False
    if risk_budgets is not None:
        if isinstance(risk_budgets, dict):
            b = np.asarray([float(risk_budgets.get(k, 1.0 / n)) for k in keys], dtype=np.float64)
        else:
            b = np.asarray(risk_budgets, dtype=np.float64).ravel()
            if b.size != n:
                b = np.full(n, 1.0 / n)
        b = np.maximum(b, 0.0)
        if float(np.sum(b)) > 0:
            b = b / float(np.sum(b))
            # Budget-weighted RP: scale ERC weights by relative budgets
            w = w * b
            s = float(np.sum(w))
            if s > 0:
                w = w / s
            budget_applied = True

    return {
        "name": "capital_risk_parity",
        "weights": {keys[i]: float(w[i]) for i in range(n)},
        "weight_vector": w.tolist(),
        "converged": bool(base.get("converged", False)),
        "iterations": int(base.get("iterations", 0)),
        "budget_applied": budget_applied,
        "base": base,
    }
