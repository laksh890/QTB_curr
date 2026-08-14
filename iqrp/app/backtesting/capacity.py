"""Capacity curves: capital → return / Sharpe / cost / drawdown."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

__all__ = ["CapacityModel", "capacity_curve", "estimate_capacity_limit"]


class CapacityModel:
    """Map capital level to cost/impact drag on a base return series.

    Default model: ``drag = impact_coef * (capital / adv) ** impact_exp * turnover``.
    """

    def __init__(
        self,
        *,
        adv: float = 1e8,
        turnover: float = 1.0,
        impact_coef: float = 0.1,
        impact_exp: float = 0.5,
        fixed_cost: float = 0.0,
        periods_per_year: float = 252.0,
    ) -> None:
        self.adv = float(adv)
        self.turnover = float(turnover)
        self.impact_coef = float(impact_coef)
        self.impact_exp = float(impact_exp)
        self.fixed_cost = float(fixed_cost)
        self.periods_per_year = float(periods_per_year)

    def per_period_cost(self, capital: float) -> float:
        """Average per-period cost fraction at ``capital``."""
        participation = float(capital) / max(self.adv, 1e-12)
        impact = self.impact_coef * (max(participation, 0.0) ** self.impact_exp)
        annual = self.fixed_cost + impact * self.turnover
        return annual / max(self.periods_per_year, 1e-12)

    def adjust_returns(self, returns: Any, capital: float) -> np.ndarray:
        """Subtract per-period cost from returns at the given capital."""
        r = as_returns(returns)
        drag = self.per_period_cost(capital)
        return r - drag


def capacity_curve(
    returns: Any,
    capital_levels: Any,
    *,
    model: CapacityModel | None = None,
    cost_fn: Callable[[float], float] | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Build a capacity curve across capital levels.

    Returns arrays for capital, expected return, Sharpe, cost, and drawdown.
    Curve length equals ``len(capital_levels)`` (must be > 1 for a meaningful curve).
    """
    levels = np.asarray(capital_levels, dtype=np.float64).reshape(-1)
    if levels.size == 0:
        raise ValueError("capital_levels must be non-empty")

    mdl = model or CapacityModel(periods_per_year=periods_per_year)
    base = as_returns(returns)

    capitals: list[float] = []
    exp_returns: list[float] = []
    sharpes: list[float] = []
    costs: list[float] = []
    drawdowns: list[float] = []

    for cap in levels.tolist():
        c = float(cap)
        if cost_fn is not None:
            drag = float(cost_fn(c))
            adj = base - drag
            cost = drag * base.size
        else:
            adj = mdl.adjust_returns(base, c)
            cost = mdl.per_period_cost(c) * base.size
        capitals.append(c)
        exp_returns.append(total_return(adj))
        sharpes.append(sharpe_ratio(adj, periods_per_year=periods_per_year))
        costs.append(float(cost))
        drawdowns.append(max_drawdown(adj))

    return {
        "name": "capacity_curve",
        "capital": np.asarray(capitals, dtype=np.float64),
        "expected_return": np.asarray(exp_returns, dtype=np.float64),
        "expected_sharpe": np.asarray(sharpes, dtype=np.float64),
        "expected_cost": np.asarray(costs, dtype=np.float64),
        "expected_drawdown": np.asarray(drawdowns, dtype=np.float64),
        "n_levels": int(levels.size),
    }


def estimate_capacity_limit(
    returns: Any,
    *,
    capital_levels: Any | None = None,
    min_sharpe: float = 0.5,
    max_drawdown: float = 0.25,
    model: CapacityModel | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Largest capital at which Sharpe and drawdown gates still pass."""
    if capital_levels is None:
        capital_levels = np.geomspace(1e6, 1e9, 16)
    curve = capacity_curve(
        returns,
        capital_levels,
        model=model,
        periods_per_year=periods_per_year,
    )
    ok = (curve["expected_sharpe"] >= float(min_sharpe)) & (
        curve["expected_drawdown"] <= float(max_drawdown)
    )
    if not np.any(ok):
        limit = 0.0
    else:
        limit = float(np.max(curve["capital"][ok]))
    return {
        "capacity_limit": limit,
        "min_sharpe": float(min_sharpe),
        "max_drawdown": float(max_drawdown),
        "curve": curve,
    }
