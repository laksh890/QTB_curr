"""Equal risk contribution capital weights via sizing.equal_risk_contribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.sizing.risk_parity import equal_risk_contribution


def equal_risk_weights(
    cov: Any,
    *,
    names: list[str] | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Call ``equal_risk_contribution`` and map to named capital weights."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    keys = names if names and len(names) == n else [f"s{i}" for i in range(n)]
    result = equal_risk_contribution(c, max_iter=max_iter, tol=tol)
    w = np.asarray(result.get("weights") or [], dtype=np.float64)
    if w.size != n:
        w = np.full(n, 1.0 / n if n else 0.0)
    return {
        "name": "equal_risk_weights",
        "weights": {keys[i]: float(w[i]) for i in range(n)},
        "weight_vector": w.tolist(),
        "converged": bool(result.get("converged", False)),
        "iterations": int(result.get("iterations", 0)),
        "component_risk_contribution": result.get("component_risk_contribution"),
        "base": result,
    }
