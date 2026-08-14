"""Performance metrics for a horizon cell — reuses accounting/performance libs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.horizon.trade_analytics import trade_frequency_report
from iqrp.app.backtesting.performance.drawdown import average_drawdown, max_drawdown
from iqrp.app.backtesting.performance.returns import annualized_return, as_returns, cagr, total_return
from iqrp.app.backtesting.performance.risk_adjusted import calmar_ratio, sharpe_ratio, sortino_ratio
from iqrp.app.backtesting.performance.trade_metrics import (
    average_holding_period,
    average_loss,
    average_win,
    expectancy,
    loss_rate,
    profit_factor,
    summarize_trades,
    win_rate,
)


def horizon_performance_metrics(
    gross_returns: Any,
    net_returns: Any | None = None,
    *,
    trades: Sequence[Mapping[str, Any]] | None = None,
    periods_per_year: float = 252.0,
    turnover: float | None = None,
) -> dict[str, Any]:
    """Minimum metric set required by the horizon research brief."""
    g = as_returns(gross_returns)
    n = as_returns(net_returns) if net_returns is not None else g
    trade_sum = summarize_trades(trades) if trades else {}
    pnls = []
    if trades:
        for t in trades:
            if "pnl" in t:
                pnls.append(float(t["pnl"]))
    pnl_arr = np.asarray(pnls, dtype=np.float64) if pnls else np.zeros(0)

    freq = trade_frequency_report(trades or [])

    metrics: dict[str, Any] = {
        "total_return_gross": float(total_return(g)),
        "total_return_net": float(total_return(n)),
        "total_return": float(total_return(n)),
        "cagr": float(cagr(n, periods_per_year=periods_per_year)) if n.size else 0.0,
        "annualized_return": float(annualized_return(n, periods_per_year=periods_per_year))
        if n.size
        else 0.0,
        "volatility": float(np.std(n, ddof=1) * np.sqrt(periods_per_year)) if n.size > 1 else 0.0,
        "sharpe": float(sharpe_ratio(n, periods_per_year=periods_per_year)),
        "gross_sharpe": float(sharpe_ratio(g, periods_per_year=periods_per_year)),
        "net_sharpe": float(sharpe_ratio(n, periods_per_year=periods_per_year)),
        "sortino": float(sortino_ratio(n, periods_per_year=periods_per_year)),
        "calmar": float(calmar_ratio(n, periods_per_year=periods_per_year)),
        "maximum_drawdown": float(max_drawdown(n)),
        "average_drawdown": float(average_drawdown(n)),
        "win_rate": float(win_rate(trades)) if trades else (float(np.mean(pnl_arr > 0)) if pnl_arr.size else 0.0),
        "loss_rate": float(loss_rate(trades)) if trades else (float(np.mean(pnl_arr < 0)) if pnl_arr.size else 0.0),
        "profit_factor": float(profit_factor(trades)) if trades else 0.0,
        "expectancy_per_trade": float(expectancy(trades)) if trades else (float(np.mean(pnl_arr)) if pnl_arr.size else 0.0),
        "average_winning_trade": float(average_win(trades)) if trades else None,
        "average_losing_trade": float(average_loss(trades)) if trades else None,
        "median_trade": float(np.median(pnl_arr)) if pnl_arr.size else None,
        "best_trade": float(np.max(pnl_arr)) if pnl_arr.size else None,
        "worst_trade": float(np.min(pnl_arr)) if pnl_arr.size else None,
        "average_holding_period": float(average_holding_period(trades))
        if trades
        else freq.get("average_holding_period_seconds"),
        "turnover": turnover,
        "trade_count": int(freq.get("total_trades", 0)),
        "long_trade_count": int(freq.get("long_trades", 0)),
        "short_trade_count": int(freq.get("short_trades", 0)),
        "trade_summary": trade_sum,
    }
    return metrics


__all__ = ["horizon_performance_metrics"]
