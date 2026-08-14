"""Scalability: how net Sharpe / returns degrade with AUM."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.alpha.economics.capacity import capacity_decay, estimate_capacity
from iqrp.app.alpha.economics.market_impact import market_impact_bps
from iqrp.app.alpha.economics.slippage import slippage_bps


def scalability_curve(
    *,
    capitals: Any,
    gross_sharpe: float,
    turnover: float,
    adv: float,
    max_participation: float = 0.1,
    vol: float = 0.01,
    impact_coeff: float = 0.1,
    base_slippage_bps: float = 1.0,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Map AUM levels → participation, cost bps, and implied net Sharpe.

    Net Sharpe approximates ``gross_sharpe - cost_drag``, where cost drag is
    annualized turnover × total_cost_bps / 1e4 scaled into Sharpe units via
    dividing by a unit-vol assumption (vol).
    """
    caps = np.asarray(capitals, dtype=np.float64).reshape(-1)
    cap_info = estimate_capacity(turnover=turnover, adv=adv, max_participation=max_participation)
    max_cap = float(cap_info["max_capital"])
    to = max(float(turnover), 1e-12)
    participation = (caps * to) / max(float(adv), 1e-12)
    impact = market_impact_bps(participation, impact_coeff=impact_coeff, vol=vol)
    slip = slippage_bps(participation, base_bps=base_slippage_bps, vol=vol)
    total_bps = np.asarray(impact, dtype=np.float64) + np.asarray(slip, dtype=np.float64)
    # Annualized return drag ≈ turnover_per_period * periods * bps/1e4
    # Here turnover is per-period; annualize
    ann_drag = to * float(periods_per_year) * (total_bps / 1e4)
    vol_ann = max(float(vol) * np.sqrt(float(periods_per_year)), 1e-12)
    net_sharpe = float(gross_sharpe) - ann_drag / vol_ann
    decay = capacity_decay(caps, max_capital=max_cap, decay_power=1.0)
    return {
        "capitals": caps,
        "participation": participation,
        "impact_bps": np.asarray(impact, dtype=np.float64),
        "slippage_bps": np.asarray(slip, dtype=np.float64),
        "total_cost_bps": total_bps,
        "net_sharpe": net_sharpe,
        "decay": decay,
        "max_capital": max_cap,
        "breakeven_capital": (
            float(caps[np.nanargmin(np.abs(net_sharpe))]) if caps.size else float("nan")
        ),
    }


def scalability_report(
    *,
    gross_sharpe: float,
    turnover: float,
    adv: float,
    max_participation: float = 0.1,
    n_points: int = 20,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience report over a log-spaced capital grid up to 2× capacity."""
    info = estimate_capacity(turnover=turnover, adv=adv, max_participation=max_participation)
    max_cap = max(float(info["max_capital"]), 1.0)
    capitals = np.geomspace(max_cap / 100.0, max_cap * 2.0, num=max(int(n_points), 3))
    curve = scalability_curve(
        capitals=capitals,
        gross_sharpe=gross_sharpe,
        turnover=turnover,
        adv=adv,
        max_participation=max_participation,
        **kwargs,
    )
    ns = np.asarray(curve["net_sharpe"], dtype=np.float64)
    viable = ns > 0
    max_viable = float(np.max(capitals[viable])) if np.any(viable) else 0.0
    return {
        **curve,
        "capacity": info,
        "max_viable_capital": max_viable,
        "gross_sharpe": float(gross_sharpe),
    }
