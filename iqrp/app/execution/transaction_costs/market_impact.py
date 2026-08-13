"""Market impact transaction cost."""

from __future__ import annotations

from typing import Any

from iqrp.app.execution.slippage.market_impact import market_impact
from iqrp.app.execution.slippage.nonlinear_impact import nonlinear_impact


def market_impact_cost(
    *,
    side: str,
    quantity: float,
    mid: float,
    adv: float = 1e6,
    volatility: float = 0.02,
    spread: float = 0.0,
    impact_coeff: float = 0.1,
    use_nonlinear: bool = False,
    nonlinear_exponent: float = 0.6,
    include_permanent: bool = False,
) -> dict[str, Any]:
    """Currency market-impact cost (qty * impact_px)."""
    if use_nonlinear:
        impact = nonlinear_impact(
            quantity=quantity,
            mid=mid,
            adv=adv,
            volatility=volatility,
            impact_coeff=impact_coeff,
            exponent=nonlinear_exponent,
        )
    else:
        impact = market_impact(
            side=side,
            quantity=quantity,
            mid=mid,
            adv=adv,
            volatility=volatility,
            spread=spread,
            impact_coeff=impact_coeff,
        )
    px = float(impact["temporary_impact"])
    if include_permanent:
        px += float(impact["permanent_impact"])
    qty = abs(float(quantity))
    return {
        "name": "market_impact_cost",
        "total": float(qty * px),
        "impact_px": px,
        "impact_bps": float(impact["slippage_bps"])
        if not include_permanent
        else float(px / max(float(mid), 1e-12) * 1e4),
        "temporary_impact": float(impact["temporary_impact"]),
        "permanent_impact": float(impact["permanent_impact"]),
        "participation": float(impact["participation"]),
        "detail": impact,
    }


__all__ = ["market_impact_cost"]
