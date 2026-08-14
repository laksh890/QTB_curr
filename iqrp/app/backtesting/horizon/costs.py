"""Cost-aware gross vs net P&L / alpha attribution for horizon research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio


def apply_cost_drag(
    gross_returns: Any,
    *,
    commission_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    turnover_per_period: Any | None = None,
    financing_bps_per_period: float = 0.0,
    impact_bps_per_period: float = 0.0,
) -> dict[str, Any]:
    """Convert gross period returns into net returns under bps cost assumptions.

    ``turnover_per_period`` is absolute weight change per bar (fraction of NAV).
    If omitted, a constant zero turnover is assumed (no trading cost).
    """
    g = as_returns(gross_returns)
    if turnover_per_period is None:
        to = np.zeros_like(g)
    else:
        to = np.asarray(turnover_per_period, dtype=np.float64).reshape(-1)
        if to.size == 1:
            to = np.full_like(g, float(to[0]))
        elif to.size != g.size:
            # align by truncation / pad
            n = min(to.size, g.size)
            tmp = np.zeros_like(g)
            tmp[:n] = to[:n]
            to = tmp

    trade_bps = float(commission_bps) + float(spread_bps) + float(slippage_bps)
    trade_cost = to * (trade_bps / 10_000.0)
    fin = np.full_like(g, float(financing_bps_per_period) / 10_000.0)
    impact = np.full_like(g, float(impact_bps_per_period) / 10_000.0)
    total_cost = trade_cost + fin + impact
    net = g - total_cost

    return {
        "gross_returns": g,
        "net_returns": net,
        "commission": float(np.sum(to * (float(commission_bps) / 10_000.0))),
        "spread_cost": float(np.sum(to * (float(spread_bps) / 10_000.0))),
        "slippage": float(np.sum(to * (float(slippage_bps) / 10_000.0))),
        "financing_borrow": float(np.sum(fin)),
        "market_impact": float(np.sum(impact)),
        "transaction_costs": float(np.sum(total_cost)),
        "gross_pnl": float(total_return(g)),
        "net_pnl": float(total_return(net)),
        "gross_alpha": float(np.mean(g)) if g.size else 0.0,
        "net_alpha": float(np.mean(net)) if net.size else 0.0,
        "cost_eroded_edge": bool(
            (float(total_return(g)) > 0 and float(total_return(net)) <= 0)
            or (
                g.size >= 2
                and sharpe_ratio(g) > 1.0
                and sharpe_ratio(net) < 0.5
            )
        ),
    }


def cost_attribution_from_trades(
    trades: Sequence[Mapping[str, Any]] | None,
    *,
    commission_key: str = "commission",
    spread_key: str = "spread",
    slippage_key: str = "slippage",
    impact_key: str = "market_impact",
    financing_key: str = "financing",
    pnl_key: str = "pnl",
) -> dict[str, Any]:
    """Aggregate cost fields from trade blotter rows when present."""
    rows = [dict(t) for t in (trades or [])]
    def _sum(k: str) -> float:
        return float(sum(float(t.get(k, 0.0) or 0.0) for t in rows))

    commissions = _sum(commission_key)
    spread = _sum(spread_key)
    slip = _sum(slippage_key)
    impact = _sum(impact_key)
    fin = _sum(financing_key)
    gross = float(sum(float(t.get(pnl_key, 0.0) or 0.0) for t in rows))
    # If trades already net of costs, reconstruct gross by adding costs back
    total_cost = commissions + spread + slip + impact + fin
    # Prefer explicit gross_pnl if present
    if rows and any("gross_pnl" in t for t in rows):
        gross = float(sum(float(t.get("gross_pnl", t.get(pnl_key, 0.0)) or 0.0) for t in rows))
        net = float(sum(float(t.get("net_pnl", t.get(pnl_key, 0.0)) or 0.0) for t in rows))
    else:
        net = gross  # assume pnl already net unless costs separate
        if total_cost and all("gross_pnl" not in t for t in rows):
            # treat pnl as net
            pass

    return {
        "gross_pnl": gross,
        "commissions": commissions,
        "spread_cost": spread,
        "slippage": slip,
        "transaction_costs": total_cost,
        "financing_borrow": fin,
        "market_impact": impact,
        "net_pnl": net if rows and any("net_pnl" in t for t in rows) else (gross - total_cost if total_cost else net),
        "n_trades": len(rows),
        "cost_eroded_edge": bool(gross > 0 and (gross - total_cost) <= 0),
    }


def gross_vs_net_sharpe(
    gross_returns: Any,
    net_returns: Any,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    g = as_returns(gross_returns)
    n = as_returns(net_returns)
    gs = sharpe_ratio(g, periods_per_year=periods_per_year)
    ns = sharpe_ratio(n, periods_per_year=periods_per_year)
    return {
        "gross_sharpe": gs,
        "net_sharpe": ns,
        "sharpe_decay": float(gs - ns),
        "cost_inefficient": bool(gs >= 1.0 and ns < 0.5),
        "note": "High gross Sharpe with collapsed net Sharpe is COST-INEFFICIENT.",
    }


__all__ = [
    "apply_cost_drag",
    "cost_attribution_from_trades",
    "gross_vs_net_sharpe",
]
