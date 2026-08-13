"""Pre-trade expected slippage estimation."""

from __future__ import annotations

from typing import Any

from iqrp.app.execution.slippage.liquidity import liquidity_slippage
from iqrp.app.execution.slippage.market_impact import market_impact
from iqrp.app.execution.slippage.model import ExecutionSlippageModel
from iqrp.app.execution.slippage.nonlinear_impact import nonlinear_impact
from iqrp.app.execution.slippage.spread import spread_slippage
from iqrp.app.execution.slippage.volatility import volatility_slippage


def estimate_slippage(
    *,
    side: str,
    quantity: float,
    mid: float,
    spread: float = 0.0,
    adv: float = 1e6,
    volatility: float = 0.02,
    liquidity: float = 1.0,
    delay_seconds: float = 0.0,
    horizon_seconds: float = 0.0,
    impact_coeff: float = 0.1,
    use_nonlinear: bool = False,
    nonlinear_exponent: float = 0.6,
    model: ExecutionSlippageModel | None = None,
) -> dict[str, Any]:
    """Estimate expected pre-trade slippage and return a structured breakdown.

    Returns a dict including ``expected_slippage_bps`` and component details.
    """
    mdl = model or ExecutionSlippageModel(impact_coeff=impact_coeff)
    br = mdl.estimate(
        side=side,
        quantity=quantity,
        mid=mid,
        spread=spread,
        adv=adv,
        volatility=volatility,
        liquidity=liquidity,
        delay_seconds=delay_seconds,
        horizon_seconds=horizon_seconds,
    )

    spread_c = spread_slippage(mid=mid, spread=spread, side=side)
    vol_c = volatility_slippage(
        mid=mid,
        volatility=volatility,
        horizon_seconds=horizon_seconds or delay_seconds,
        delay_seconds=delay_seconds,
    )
    liq_c = liquidity_slippage(
        mid=mid, quantity=quantity, adv=adv, liquidity=liquidity
    )
    if use_nonlinear:
        impact_c = nonlinear_impact(
            quantity=quantity,
            mid=mid,
            adv=adv,
            volatility=volatility,
            impact_coeff=impact_coeff,
            exponent=nonlinear_exponent,
        )
    else:
        impact_c = market_impact(
            side=side,
            quantity=quantity,
            mid=mid,
            adv=adv,
            volatility=volatility,
            spread=spread,
            impact_coeff=impact_coeff,
        )

    components = {
        "spread": float(spread_c["slippage"]),
        "volatility": float(vol_c["slippage"]),
        "liquidity": float(liq_c["slippage"]),
        "temporary_impact": float(impact_c["temporary_impact"]),
        "permanent_impact": float(impact_c["permanent_impact"]),
        "delay": float(br.delay),
    }
    # Prefer composite model total (avoids double-counting if summing all)
    expected_px = float(br.total)
    expected_bps = float(br.total_bps)
    mid_f = max(float(mid), 1e-12)
    notional = abs(float(quantity)) * mid_f

    return {
        "name": "estimate_slippage",
        "side": str(side).lower(),
        "quantity": abs(float(quantity)),
        "mid": mid_f,
        "expected_slippage": expected_px,
        "expected_slippage_bps": expected_bps,
        "expected_slippage_notional": expected_px * abs(float(quantity)),
        "notional": notional,
        "components": components,
        "breakdown": br.to_dict(),
        "spread_detail": spread_c,
        "impact_detail": impact_c,
        "volatility_detail": vol_c,
        "liquidity_detail": liq_c,
        "participation": abs(float(quantity)) / max(float(adv), 1e-12),
    }


__all__ = ["estimate_slippage"]
