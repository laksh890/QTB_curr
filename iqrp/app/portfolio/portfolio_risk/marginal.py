"""Marginal risk contribution wrapper."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.portfolio.portfolio_risk import marginal_risk_contribution as _mrc


def marginal_risk_contribution(weights: Any, cov: Any) -> dict[str, Any]:
    """Wrap ``iqrp.app.risk.portfolio.portfolio_risk.marginal_risk_contribution``."""
    out = dict(_mrc(weights, cov))
    out["source"] = "iqrp.app.risk.portfolio.portfolio_risk.marginal_risk_contribution"
    return out


# Convenience alias
marginal = marginal_risk_contribution
