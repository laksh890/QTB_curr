"""Component risk contribution wrapper."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.portfolio.portfolio_risk import component_risk_contribution as _crc


def component_risk_contribution(weights: Any, cov: Any) -> dict[str, Any]:
    """Wrap ``iqrp.app.risk.portfolio.portfolio_risk.component_risk_contribution``."""
    out = dict(_crc(weights, cov))
    out["source"] = "iqrp.app.risk.portfolio.portfolio_risk.component_risk_contribution"
    return out


component = component_risk_contribution
