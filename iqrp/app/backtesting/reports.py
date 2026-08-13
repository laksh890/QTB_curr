"""Backtest reporting: performance, risk, drawdown, trade, exposure, attribution,
cost, execution, scenario, capacity, sensitivity, and scorecard summaries.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from iqrp.app.backtesting.performance import (
    build_scorecard,
    compare_to_benchmark,
    full_attribution,
    stability_report,
    summarize_drawdown,
    summarize_exposure,
    summarize_returns,
    summarize_risk_adjusted,
    summarize_tail,
    summarize_trades,
)
from iqrp.app.backtesting.performance.returns import as_returns, wealth_index
from iqrp.app.backtesting.serializer import to_jsonable

__all__ = [
    "performance_report",
    "risk_report",
    "drawdown_report",
    "trade_report",
    "exposure_report",
    "attribution_report",
    "cost_report",
    "execution_report",
    "scenario_report",
    "capacity_report",
    "sensitivity_report",
    "scorecard_report",
    "full_report",
]


def performance_report(returns: Any, **kwargs: Any) -> dict[str, Any]:
    r = as_returns(returns)
    return {
        "name": "performance",
        "returns": summarize_returns(r, **{k: v for k, v in kwargs.items() if k in ("periods_per_year",)}),
        "risk_adjusted": summarize_risk_adjusted(r, **{k: v for k, v in kwargs.items() if k in ("risk_free", "periods_per_year")}),
        "equity": wealth_index(r).tolist(),
    }


def risk_report(returns: Any, **kwargs: Any) -> dict[str, Any]:
    r = as_returns(returns)
    return {
        "name": "risk",
        "risk_adjusted": summarize_risk_adjusted(r, **kwargs),
        "tail": summarize_tail(r),
        "stability": stability_report(r),
    }


def drawdown_report(returns: Any) -> dict[str, Any]:
    return {"name": "drawdown", **summarize_drawdown(returns)}


def trade_report(trades: Any = None, *, positions: Any | None = None) -> dict[str, Any]:
    return {"name": "trades", **summarize_trades(trades, positions=positions)}


def exposure_report(exposures: Any) -> dict[str, Any]:
    return {"name": "exposure", **summarize_exposure(exposures)}


def attribution_report(
    returns: Any,
    *,
    factors: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if factors is None:
        return {
            "name": "attribution",
            "note": "no factors provided",
            "total": float(np.sum(as_returns(returns))),
        }
    return {
        "name": "attribution",
        **full_attribution(
            returns=returns,
            factor_returns=factors,
            **kwargs,
        ),
    }


def cost_report(costs: Any) -> dict[str, Any]:
    c = as_returns(costs) if costs is not None else np.asarray([], dtype=np.float64)
    return {
        "name": "costs",
        "total": float(np.sum(np.abs(c))) if c.size else 0.0,
        "mean": float(np.mean(np.abs(c))) if c.size else 0.0,
        "series": c.tolist(),
    }


def execution_report(fills: Any = None, *, latency: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fills_list = list(fills or [])
    return {
        "name": "execution",
        "n_fills": len(fills_list),
        "fills": to_jsonable(fills_list[:100]),
        "latency": dict(latency or {}),
    }


def scenario_report(scenario_results: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"name": "scenarios", "results": to_jsonable(dict(scenario_results or {}))}


def capacity_report(capacity_results: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"name": "capacity", "results": to_jsonable(dict(capacity_results or {}))}


def sensitivity_report(sensitivity_results: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"name": "sensitivity", "results": to_jsonable(dict(sensitivity_results or {}))}


def scorecard_report(
    returns: Any,
    *,
    positions: Any | None = None,
    costs: Any | None = None,
    oos_returns: Any | None = None,
    capacity: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    sc = build_scorecard(
        returns,
        positions=positions,
        costs=costs,
        oos_returns=oos_returns,
        capacity=capacity,
        **kwargs,
    )
    return {"name": "scorecard", **sc.to_dict()}


def full_report(result: Any) -> dict[str, Any]:
    """Assemble a multi-section report from a :class:`BacktestResult`."""
    returns = getattr(result, "returns", None)
    trades = getattr(result, "trades", None)
    exposures = getattr(result, "exposures", None)
    costs = getattr(result, "costs", None)
    attribution = getattr(result, "attribution", None)
    reports: dict[str, Any] = {
        "experiment_id": getattr(result, "experiment_id", None),
        "state": str(getattr(result, "state", "")),
    }
    if returns is not None:
        reports["performance"] = performance_report(returns)
        reports["risk"] = risk_report(returns)
        reports["drawdown"] = drawdown_report(returns)
        reports["scorecard"] = scorecard_report(
            returns,
            positions=exposures,
            costs=costs,
            oos_returns=getattr(result, "oos_returns", None),
        )
    if trades is not None:
        reports["trades"] = trade_report(trades, positions=exposures)
    if exposures is not None:
        reports["exposure"] = exposure_report(exposures)
    if costs is not None:
        reports["costs"] = cost_report(costs)
    if attribution is not None:
        reports["attribution"] = {"name": "attribution", **to_jsonable(attribution)}
    bench = getattr(result, "benchmark_returns", None)
    if bench is not None and returns is not None:
        reports["benchmark"] = compare_to_benchmark(returns, kind="custom", benchmark=bench)
    return to_jsonable(reports)
