"""Percentage risk contribution and volatility contribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.portfolio_risk.component import component_risk_contribution
from iqrp.app.portfolio.portfolio_risk.marginal import marginal_risk_contribution
from iqrp.app.risk.portfolio.portfolio_risk import portfolio_volatility


def percentage_risk_contribution(weights: Any, cov: Any) -> dict[str, Any]:
    """PRC_i = CRC_i / sum(CRC) — sums to 1 when portfolio vol > 0."""
    crc = component_risk_contribution(weights, cov)
    values = np.asarray(crc["values"], dtype=np.float64)
    sigma = float(crc.get("portfolio_volatility", np.sum(values)))
    if abs(sigma) <= 1e-12:
        pct = np.zeros_like(values)
    else:
        pct = values / sigma
    return {
        "name": "percentage_risk_contribution",
        "values": pct.tolist(),
        "percent": (pct * 100.0).tolist(),
        "component": values.tolist(),
        "portfolio_volatility": abs(sigma),
    }


def volatility_contribution(weights: Any, cov: Any) -> dict[str, Any]:
    """Volatility contribution equals component risk contribution (Euler allocation)."""
    crc = component_risk_contribution(weights, cov)
    mrc = marginal_risk_contribution(weights, cov)
    vol = portfolio_volatility(weights, cov)
    return {
        "name": "volatility_contribution",
        "values": list(crc["values"]),
        "marginal": list(mrc["values"]),
        "portfolio_volatility": float(vol.value),
        "percent": list(crc.get("percent", [])),
    }


def risk_contribution(weights: Any, cov: Any) -> dict[str, Any]:
    """Bundle of PRC + volatility contribution."""
    prc = percentage_risk_contribution(weights, cov)
    vc = volatility_contribution(weights, cov)
    return {
        "name": "risk_contribution",
        "percentage_risk_contribution": prc,
        "volatility_contribution": vc,
        "portfolio_volatility": prc["portfolio_volatility"],
    }
