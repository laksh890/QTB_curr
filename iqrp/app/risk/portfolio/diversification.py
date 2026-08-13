"""Diversification ratio."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_weights


def diversification_ratio(
    weights: Any,
    cov: Any,
) -> RiskMeasure:
    """Diversification ratio = weighted avg vol / portfolio vol."""
    w = as_weights(weights)
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be a square matrix")
    n = c.shape[0]
    w = as_weights(w, n=n)
    vols = np.sqrt(np.maximum(np.diag(c), 0.0))
    port_var = float(w @ c @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    weighted_vol = float(np.abs(w) @ vols)
    if port_vol <= 1e-12:
        ratio = 1.0 if weighted_vol <= 1e-12 else weighted_vol / 1e-12
    else:
        ratio = weighted_vol / port_vol
    return RiskMeasure(
        name="diversification_ratio",
        value=float(ratio),
        unit="ratio",
        method="weighted_vol/portfolio_vol",
        parameters={"n_assets": n, "portfolio_vol": port_vol, "weighted_avg_vol": weighted_vol},
    )
