"""Portfolio volatility and risk contribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_weights
from iqrp.app.risk.portfolio.concentration import concentration_risk
from iqrp.app.risk.portfolio.diversification import diversification_ratio
from iqrp.app.risk.portfolio.exposure import exposure_summary


def portfolio_volatility(weights: Any, cov: Any) -> RiskMeasure:
    w = as_weights(weights)
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    w = as_weights(w, n=n)
    var = float(w @ c @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    return RiskMeasure(
        name="portfolio_volatility",
        value=vol,
        unit="volatility",
        method="sqrt_w_cov_w",
        parameters={"n_assets": n, "variance": var},
    )


def marginal_risk_contribution(weights: Any, cov: Any) -> dict[str, Any]:
    """Marginal contribution to risk: d(sigma)/d(w_i) = (Cov w)_i / sigma."""
    w = as_weights(weights)
    c = np.asarray(cov, dtype=np.float64)
    n = c.shape[0]
    w = as_weights(w, n=n)
    sigma = portfolio_volatility(w, c).value
    cov_w = c @ w
    if sigma <= 1e-12:
        mrc = np.zeros(n, dtype=np.float64)
    else:
        mrc = cov_w / sigma
    return {
        "name": "marginal_risk_contribution",
        "values": mrc.tolist(),
        "portfolio_volatility": sigma,
    }


def component_risk_contribution(weights: Any, cov: Any) -> dict[str, Any]:
    """Component risk contribution: w_i * MRC_i (sums to portfolio vol)."""
    w = as_weights(weights)
    c = np.asarray(cov, dtype=np.float64)
    n = c.shape[0]
    w = as_weights(w, n=n)
    mrc = np.asarray(marginal_risk_contribution(w, c)["values"], dtype=np.float64)
    crc = w * mrc
    sigma = float(np.sum(crc))
    pct = (crc / sigma).tolist() if abs(sigma) > 1e-12 else [0.0] * n
    return {
        "name": "component_risk_contribution",
        "values": crc.tolist(),
        "percent": pct,
        "portfolio_volatility": abs(sigma),
    }


def portfolio_risk(weights: Any, cov: Any) -> dict[str, Any]:
    """Aggregator of core portfolio risk diagnostics."""
    vol = portfolio_volatility(weights, cov)
    return {
        "name": "portfolio_risk",
        "volatility": vol.to_dict(),
        "marginal_risk_contribution": marginal_risk_contribution(weights, cov),
        "component_risk_contribution": component_risk_contribution(weights, cov),
        "exposure": exposure_summary(weights),
        "concentration": concentration_risk(weights),
        "diversification_ratio": diversification_ratio(weights, cov).to_dict(),
    }
