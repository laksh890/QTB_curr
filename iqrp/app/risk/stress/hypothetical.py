"""Hypothetical covariance-based stress."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_weights
from iqrp.app.risk.stress.scenarios import ScenarioSpec, apply_shock


def hypothetical_stress(
    weights: Any,
    cov: Any,
    shocks: Any,
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Apply hypothetical shocks and report PnL plus quadratic risk context.

    ``shocks`` may be a ScenarioSpec, dict, or vector. Covariance is used to
    report portfolio variance under stressed weights context (diagnostic only);
    the primary loss is linear w·shock (no look-ahead).
    """
    shock_result = apply_shock(weights, shocks, names=names)
    w = as_weights(shock_result["weights"])
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    w = as_weights(w, n=n)
    port_var = float(w @ c @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))

    scenario_name = shock_result["scenario"]
    if isinstance(shocks, ScenarioSpec):
        scenario_name = shocks.name

    return {
        "name": "hypothetical_stress",
        "scenario": scenario_name,
        "pnl": shock_result["pnl"],
        "loss": shock_result["loss"],
        "portfolio_volatility": port_vol,
        "shocks": shock_result["shocks"],
        "measures": {
            "hypothetical_loss": RiskMeasure(
                name="hypothetical_stress_loss",
                value=float(shock_result["loss"]),
                unit="return",
                method="hypothetical",
                parameters={"scenario": scenario_name, "portfolio_vol": port_vol},
            ).to_dict(),
        },
    }
