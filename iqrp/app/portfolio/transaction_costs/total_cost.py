"""Aggregate transaction cost for weight transitions / trade lists."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.transaction_costs.commissions import commission_cost
from iqrp.app.portfolio.transaction_costs.market_impact import market_impact_cost
from iqrp.app.portfolio.transaction_costs.slippage import slippage_cost
from iqrp.app.portfolio.transaction_costs.spread import spread_cost


def _weight_delta(weights_old: Any, weights_new: Any) -> np.ndarray:
    a = np.asarray(weights_old, dtype=np.float64).reshape(-1)
    b = np.asarray(weights_new, dtype=np.float64).reshape(-1)
    n = max(a.size, b.size)
    aa = np.zeros(n, dtype=np.float64)
    bb = np.zeros(n, dtype=np.float64)
    aa[: a.size] = a
    bb[: b.size] = b
    return bb - aa


def total_transaction_cost(
    weights_old: Any,
    weights_new: Any,
    *,
    capital: float = 1.0,
    prices: Any | None = None,
    adv: Any | None = None,
    spreads: Any | None = None,
    vols: Any | None = None,
    commission_bps: float = 1.0,
    commission_per_share: float = 0.0,
    min_commission: float = 0.0,
    slippage_bps: float = 0.0,
    impact_coeff: float = 0.1,
    participation_coeff: float = 0.0,
    half_spread: bool = True,
    include_slippage: bool = True,
    include_impact: bool = True,
) -> dict[str, Any]:
    """Sum commission + spread + slippage + market impact for a rebalance.

    Returns structured dict with ``total``, ``components``, and ``turnover``.
    """
    delta = _weight_delta(weights_old, weights_new)
    trades = np.abs(delta)
    turnover = 0.5 * float(np.sum(trades))
    cap = max(float(capital), 0.0)

    commissions = commission_cost(
        delta,
        capital=cap,
        prices=prices,
        commission_bps=commission_bps,
        commission_per_share=commission_per_share,
        min_commission=min_commission,
    )
    spreads_c = spread_cost(
        delta,
        capital=cap,
        spreads=spreads,
        half_spread=half_spread,
    )

    components: dict[str, Any] = {
        "commissions": commissions,
        "spread": spreads_c,
    }
    total = float(commissions["total"]) + float(spreads_c["total"])

    if include_slippage:
        slip = slippage_cost(
            delta,
            capital=cap,
            vols=vols,
            adv=adv,
            prices=prices,
            slippage_bps=slippage_bps,
            participation_coeff=participation_coeff,
        )
        components["slippage"] = slip
        total += float(slip["total"])

    if include_impact:
        impact = market_impact_cost(
            delta,
            capital=cap,
            adv=adv,
            prices=prices,
            vols=vols,
            impact_coeff=impact_coeff,
        )
        components["market_impact"] = impact
        total += float(impact["total"])

    per_asset = np.zeros(delta.size, dtype=np.float64)
    for comp in components.values():
        pa = np.asarray(comp.get("per_asset", []), dtype=np.float64)
        if pa.size:
            m = min(per_asset.size, pa.size)
            per_asset[:m] += pa[:m]

    return {
        "name": "total_transaction_cost",
        "total": float(total),
        "components": components,
        "turnover": turnover,
        "trade_weights": delta.tolist(),
        "per_asset": per_asset.tolist(),
        "parameters": {
            "capital": cap,
            "commission_bps": float(commission_bps),
            "impact_coeff": float(impact_coeff),
            "half_spread": bool(half_spread),
        },
        "total_bps": float(total / cap * 1e4) if cap > 0 else 0.0,
    }


# Alias matching common naming in the architecture doc
total_cost = total_transaction_cost


def trade_list_cost(
    trades: Any,
    *,
    capital: float = 1.0,
    prices: Any | None = None,
    adv: Any | None = None,
    spreads: Any | None = None,
    vols: Any | None = None,
    commission_bps: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Cost a explicit trade list (weight deltas)."""
    t = np.asarray(trades, dtype=np.float64).reshape(-1)
    zeros = np.zeros_like(t)
    return total_transaction_cost(
        zeros,
        t,
        capital=capital,
        prices=prices,
        adv=adv,
        spreads=spreads,
        vols=vols,
        commission_bps=commission_bps,
        **kwargs,
    )
