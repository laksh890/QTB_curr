"""Turnover diagnostics for horizon research."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.trade_metrics import turnover as period_turnover


def turnover_report(
    weights_or_positions: Any,
    *,
    periods_per_day: float = 1.0,
    trading_days_per_year: float = 252.0,
    net_alpha: float | None = None,
    net_pnl: float | None = None,
) -> dict[str, Any]:
    """Daily / weekly / monthly / annualized turnover and efficiency ratios."""
    w = np.asarray(weights_or_positions, dtype=np.float64)
    if w.ndim == 1:
        # treat as single-asset position series
        to_series = np.abs(np.diff(w, prepend=w[:1] if w.size else 0.0))
        # first bar: |w0|
        if w.size:
            to_series[0] = abs(float(w[0]))
        mean_period = float(np.mean(to_series)) if to_series.size else 0.0
    else:
        mean_period = float(period_turnover(w))

    ppd = max(float(periods_per_day), 1e-12)
    daily = mean_period * ppd
    weekly = daily * 5.0
    monthly = daily * 21.0
    annual = daily * float(trading_days_per_year)

    out: dict[str, Any] = {
        "mean_turnover_per_period": mean_period,
        "daily_turnover": daily,
        "weekly_turnover": weekly,
        "monthly_turnover": monthly,
        "annualized_turnover": annual,
    }
    if net_alpha is not None and abs(annual) > 1e-15:
        out["turnover_per_unit_net_alpha"] = annual / float(net_alpha) if abs(float(net_alpha)) > 1e-15 else None
    else:
        out["turnover_per_unit_net_alpha"] = None
    if net_pnl is not None and abs(annual) > 1e-15:
        out["pnl_per_unit_turnover"] = float(net_pnl) / annual
    else:
        out["pnl_per_unit_turnover"] = None
    out["note"] = "Turnover distinguishes useful HF alpha from excessive trading."
    return out


__all__ = ["turnover_report"]
