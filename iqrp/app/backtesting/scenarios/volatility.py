"""Volatility shock scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

__all__ = ["apply_volatility_shock", "run_volatility_scenario"]


def apply_volatility_shock(
    returns: Any,
    *,
    scale: float = 1.5,
    shift_mean: bool = False,
) -> dict[str, Any]:
    """Scale return deviations from the mean by ``scale`` (vol multiplier)."""
    r = np.asarray(returns, dtype=np.float64)
    s = float(scale)
    if r.ndim == 1:
        mu = float(np.mean(r)) if r.size else 0.0
        if shift_mean:
            stressed = r * s
        else:
            stressed = mu + (r - mu) * s
    else:
        mu = np.mean(r, axis=0, keepdims=True)
        if shift_mean:
            stressed = r * s
        else:
            stressed = mu + (r - mu) * s
    return {
        "name": "volatility",
        "kind": "volatility",
        "scale": s,
        "returns": stressed,
    }


def run_volatility_scenario(
    returns: Any,
    *,
    scales: list[float] | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Grid of volatility scales with performance summaries."""
    grid = [0.5, 1.0, 1.5, 2.0] if scales is None else list(scales)
    results = []
    for s in grid:
        out = apply_volatility_shock(returns, scale=float(s))
        rr = as_returns(out["returns"] if np.asarray(out["returns"]).ndim == 1 else np.mean(out["returns"], axis=1))
        results.append(
            {
                "scale": float(s),
                "total_return": total_return(rr),
                "sharpe": sharpe_ratio(rr, periods_per_year=periods_per_year),
                "max_drawdown": max_drawdown(rr),
                "realized_vol": float(np.std(rr, ddof=1) * np.sqrt(periods_per_year))
                if rr.size > 1
                else 0.0,
            }
        )
    return {"name": "volatility_grid", "kind": "volatility", "results": results}
