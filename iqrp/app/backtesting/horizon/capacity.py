"""Model-based capacity scenarios for horizon research."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.capacity import CapacityModel, capacity_curve
from iqrp.app.backtesting.horizon.types import DEFAULT_CAPITAL_LEVELS
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio


def capacity_scenario_report(
    returns: Any,
    *,
    capital_levels: Sequence[float] | None = None,
    turnover: float = 1.0,
    adv: float = 1e8,
    impact_coef: float = 0.1,
    impact_exp: float = 0.5,
    periods_per_year: float = 252.0,
    baseline_capital: float | None = None,
) -> dict[str, Any]:
    """ESTIMATED / MODEL-BASED capacity curve — not claimed market capacity."""
    levels = list(capital_levels or DEFAULT_CAPITAL_LEVELS)
    model = CapacityModel(
        adv=float(adv),
        turnover=float(turnover),
        impact_coef=float(impact_coef),
        impact_exp=float(impact_exp),
        periods_per_year=float(periods_per_year),
    )
    curve = capacity_curve(returns, levels, model=model, periods_per_year=periods_per_year)
    base = as_returns(returns)
    base_gross = float(total_return(base))
    base_sharpe = float(sharpe_ratio(base, periods_per_year=periods_per_year))
    base_cap = float(baseline_capital if baseline_capital is not None else levels[0])

    rows: list[dict[str, Any]] = []
    for i, cap in enumerate(levels):
        rows.append(
            {
                "capital": float(cap),
                "gross_return": base_gross,
                "net_return": float(curve["expected_return"][i]),
                "net_sharpe": float(curve["expected_sharpe"][i]),
                "estimated_impact_cost": float(curve["expected_cost"][i]),
                "max_drawdown": float(curve["expected_drawdown"][i]),
                "turnover": float(turnover),
                "capacity_degradation": float(
                    max(0.0, base_sharpe - float(curve["expected_sharpe"][i]))
                ),
                "label": "ESTIMATED / MODEL-BASED",
            }
        )

    # Degradation vs smallest capital
    sharpes = [r["net_sharpe"] for r in rows]
    degradation = {
        "baseline_capital": base_cap,
        "baseline_net_sharpe": sharpes[0] if sharpes else None,
        "max_capital": float(levels[-1]) if levels else None,
        "max_capital_net_sharpe": sharpes[-1] if sharpes else None,
        "sharpe_degradation": float(sharpes[0] - sharpes[-1]) if len(sharpes) >= 2 else 0.0,
    }

    return {
        "label": "ESTIMATED / MODEL-BASED",
        "disclaimer": (
            "Capacity figures are model-based estimates using configurable impact assumptions. "
            "They are NOT claimed actual market capacity unless supporting data justifies that."
        ),
        "scenarios": rows,
        "degradation": degradation,
        "model": {
            "adv": float(adv),
            "turnover": float(turnover),
            "impact_coef": float(impact_coef),
            "impact_exp": float(impact_exp),
        },
    }


__all__ = ["capacity_scenario_report"]
