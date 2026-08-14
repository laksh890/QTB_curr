"""Pre-trade cost estimation and post-trade TCA / IS attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iqrp.app.execution.transaction_costs.borrow_cost import borrow_cost
from iqrp.app.execution.transaction_costs.commissions import commission_cost
from iqrp.app.execution.transaction_costs.exchange_fees import exchange_fees
from iqrp.app.execution.transaction_costs.financing import financing_cost
from iqrp.app.execution.transaction_costs.market_impact import market_impact_cost
from iqrp.app.execution.transaction_costs.slippage import slippage_cost
from iqrp.app.execution.transaction_costs.spread import spread_cost


def _side_is_short(side: str) -> bool:
    return str(side).strip().lower() in {"sell", "short", "s"}


def _fill_vwap(fills: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    notionals = 0.0
    qty = 0.0
    for f in fills:
        q = abs(float(f.get("quantity", f.get("qty", 0.0))))
        p = float(f.get("price", f.get("fill_price", 0.0)))
        notionals += q * p
        qty += q
    if qty <= 0.0:
        return 0.0, 0.0
    return notionals / qty, qty


def pre_trade_cost_estimate(
    *,
    side: str,
    quantity: float,
    mid: float,
    spread: float = 0.0,
    adv: float = 1e6,
    volatility: float = 0.02,
    liquidity: float = 1.0,
    commission_bps: float = 1.0,
    commission_per_share: float = 0.0,
    fee_bps: float = 0.3,
    fee_per_share: float = 0.0,
    impact_coeff: float = 0.1,
    financing_rate: float = 0.0,
    financing_days: float = 0.0,
    borrow_rate: float = 0.0,
    borrow_days: float = 0.0,
    fx_cost_bps: float = 0.0,
    include_impact_in_slippage: bool = True,
    **slippage_kwargs: Any,
) -> dict[str, Any]:
    """Estimate expected total transaction cost before execution.

    Returns a dict with ``total_cost`` and component breakdown.
    """
    qty = abs(float(quantity))
    mid_f = max(float(mid), 1e-12)
    notional = qty * mid_f

    commissions = commission_cost(
        quantity=qty,
        price=mid_f,
        commission_bps=commission_bps,
        commission_per_share=commission_per_share,
        side=side,
    )
    fees = exchange_fees(
        quantity=qty,
        price=mid_f,
        fee_bps=fee_bps,
        fee_per_share=fee_per_share,
    )
    spreads = spread_cost(quantity=qty, mid=mid_f, spread=spread, side=side)
    slip = slippage_cost(
        side=side,
        quantity=qty,
        mid=mid_f,
        spread=spread,
        adv=adv,
        volatility=volatility,
        liquidity=liquidity,
        impact_coeff=impact_coeff,
        **slippage_kwargs,
    )
    impact = market_impact_cost(
        side=side,
        quantity=qty,
        mid=mid_f,
        adv=adv,
        volatility=volatility,
        spread=spread,
        impact_coeff=impact_coeff,
    )
    fin = financing_cost(notional=notional, rate=financing_rate, days=financing_days)
    borrow = borrow_cost(
        notional=notional,
        borrow_rate=borrow_rate,
        days=borrow_days,
        is_short=_side_is_short(side),
    )
    fx = notional * max(float(fx_cost_bps), 0.0) / 1e4

    # Avoid double-counting impact already inside slippage composite
    impact_total = 0.0 if include_impact_in_slippage else float(impact["total"])
    # Spread is also inside slippage model — when using composite slippage,
    # charge commissions/fees/financing/borrow + slippage, not raw spread again.
    components = {
        "commissions": float(commissions["total"]),
        "exchange_fees": float(fees["total"]),
        "slippage": float(slip["total"]),
        "market_impact": impact_total,
        "financing": float(fin["total"]),
        "borrow": float(borrow["total"]),
        "fx": float(fx),
        # Informational (not added when include_impact_in_slippage):
        "spread_informational": float(spreads["total"]),
        "impact_informational": float(impact["total"]),
    }
    total = (
        components["commissions"]
        + components["exchange_fees"]
        + components["slippage"]
        + components["market_impact"]
        + components["financing"]
        + components["borrow"]
        + components["fx"]
    )
    return {
        "name": "pre_trade_cost_estimate",
        "side": str(side).lower(),
        "quantity": qty,
        "mid": mid_f,
        "notional": float(notional),
        "total_cost": float(total),
        "total_cost_bps": float(total / notional * 1e4) if notional > 0 else 0.0,
        "expected_slippage": float(slip["slippage_px"]),
        "expected_slippage_bps": float(slip["slippage_bps"]),
        "expected_market_impact": float(impact["impact_px"]),
        "expected_market_impact_bps": float(impact["impact_bps"]),
        "expected_total_cost": float(total),
        "components": components,
        "details": {
            "commissions": commissions,
            "exchange_fees": fees,
            "spread": spreads,
            "slippage": slip,
            "market_impact": impact,
            "financing": fin,
            "borrow": borrow,
        },
    }


def post_trade_cost_analysis(
    fills: Sequence[Mapping[str, Any]],
    *,
    side: str,
    arrival_price: float,
    decision_price: float | None = None,
    mid: float | None = None,
    spread: float = 0.0,
    adv: float = 1e6,
    volatility: float = 0.02,
    commission_bps: float = 1.0,
    commission_per_share: float = 0.0,
    fee_bps: float = 0.3,
    fee_per_share: float = 0.0,
    financing_rate: float = 0.0,
    financing_days: float = 0.0,
    borrow_rate: float = 0.0,
    borrow_days: float = 0.0,
    fx_cost_bps: float = 0.0,
    parent_quantity: float | None = None,
    benchmark_vwap: float | None = None,
    benchmark_twap: float | None = None,
) -> dict[str, Any]:
    """Post-trade TCA with implementation-shortfall attribution.

    Implementation shortfall (currency) ≈ side-signed (exec − decision) * qty
    plus explicit fees/commissions. Attribution splits delay / trading / fees.
    """
    vwap, filled_qty = _fill_vwap(fills)
    arr = max(float(arrival_price), 1e-12)
    decision = float(decision_price) if decision_price is not None else arr
    mid_f = float(mid) if mid is not None else arr
    buy = not _side_is_short(side)

    if buy:
        arrival_slip_px = vwap - arr if vwap > 0 else 0.0
        decision_slip_px = vwap - decision if vwap > 0 else 0.0
        delay_px = mid_f - decision  # price moved before trading vs decision
        trading_px = vwap - mid_f if vwap > 0 else 0.0
    else:
        arrival_slip_px = arr - vwap if vwap > 0 else 0.0
        decision_slip_px = decision - vwap if vwap > 0 else 0.0
        delay_px = decision - mid_f
        trading_px = mid_f - vwap if vwap > 0 else 0.0

    # Fees on fills
    commissions = commission_cost(
        quantity=filled_qty,
        price=vwap if vwap > 0 else mid_f,
        commission_bps=commission_bps,
        commission_per_share=commission_per_share,
        side=side,
    )
    fees = exchange_fees(
        quantity=filled_qty,
        price=vwap if vwap > 0 else mid_f,
        fee_bps=fee_bps,
        fee_per_share=fee_per_share,
    )
    notional = filled_qty * (vwap if vwap > 0 else mid_f)
    fin = financing_cost(notional=notional, rate=financing_rate, days=financing_days)
    borrow = borrow_cost(
        notional=notional,
        borrow_rate=borrow_rate,
        days=borrow_days,
        is_short=_side_is_short(side),
    )
    fx = notional * max(float(fx_cost_bps), 0.0) / 1e4
    spread_c = spread_cost(quantity=filled_qty, mid=mid_f, spread=spread, side=side)

    realized_slippage = arrival_slip_px * filled_qty
    realized_impact = max(trading_px, 0.0) * filled_qty
    fee_total = (
        float(commissions["total"])
        + float(fees["total"])
        + float(fin["total"])
        + float(borrow["total"])
        + float(fx)
    )
    is_currency = decision_slip_px * filled_qty + fee_total
    # Opportunity cost on unfilled residual vs decision
    parent = float(parent_quantity) if parent_quantity is not None else filled_qty
    residual = max(parent - filled_qty, 0.0)
    # Mark residual at current mid vs decision (opportunity)
    if buy:
        opp_px = max(mid_f - decision, 0.0)
    else:
        opp_px = max(decision - mid_f, 0.0)
    opportunity = opp_px * residual

    attribution = {
        "delay_cost": float(delay_px * filled_qty),
        "trading_cost": float(trading_px * filled_qty),
        "spread_cost": float(spread_c["total"]),
        "commissions": float(commissions["total"]),
        "exchange_fees": float(fees["total"]),
        "financing": float(fin["total"]),
        "borrow": float(borrow["total"]),
        "fx": float(fx),
        "opportunity_cost": float(opportunity),
    }

    # Benchmark slips
    def _bench_bps(bench: float | None) -> float | None:
        if bench is None or vwap <= 0:
            return None
        if buy:
            return float((vwap - float(bench)) / max(float(bench), 1e-12) * 1e4)
        return float((float(bench) - vwap) / max(float(bench), 1e-12) * 1e4)

    total_realized = realized_slippage + fee_total
    return {
        "name": "post_trade_cost_analysis",
        "side": str(side).lower(),
        "vwap": float(vwap),
        "filled_quantity": float(filled_qty),
        "parent_quantity": float(parent),
        "residual_quantity": float(residual),
        "arrival_price": arr,
        "decision_price": float(decision),
        "mid": mid_f,
        "realized_cost": float(total_realized),
        "realized_cost_bps": float(total_realized / notional * 1e4) if notional > 0 else 0.0,
        "realized_slippage": float(realized_slippage),
        "realized_slippage_bps": float(arrival_slip_px / arr * 1e4) if arr > 0 else 0.0,
        "realized_market_impact": float(realized_impact),
        "implementation_shortfall": float(is_currency + opportunity),
        "implementation_shortfall_bps": float(
            (is_currency + opportunity) / max(parent * decision, 1e-12) * 1e4
        ),
        "cost_attribution": attribution,
        "benchmarks": {
            "arrival_slippage_bps": float(arrival_slip_px / arr * 1e4) if arr > 0 else 0.0,
            "decision_slippage_bps": float(decision_slip_px / max(decision, 1e-12) * 1e4),
            "vwap_slippage_bps": _bench_bps(benchmark_vwap),
            "twap_slippage_bps": _bench_bps(benchmark_twap),
        },
        "components": {
            "commissions": float(commissions["total"]),
            "exchange_fees": float(fees["total"]),
            "spread": float(spread_c["total"]),
            "slippage": float(realized_slippage),
            "market_impact": float(realized_impact),
            "financing": float(fin["total"]),
            "borrow": float(borrow["total"]),
            "fx": float(fx),
            "opportunity": float(opportunity),
        },
        "fill_rate": float(filled_qty / parent) if parent > 0 else 0.0,
        "details": {
            "commissions": commissions,
            "exchange_fees": fees,
            "spread": spread_c,
            "financing": fin,
            "borrow": borrow,
        },
        "adv": float(adv),
        "volatility": float(volatility),
    }


# Aliases matching architecture naming
pre_trade_estimate = pre_trade_cost_estimate
post_trade_analyze = post_trade_cost_analysis


__all__ = [
    "post_trade_analyze",
    "post_trade_cost_analysis",
    "pre_trade_cost_estimate",
    "pre_trade_estimate",
]
