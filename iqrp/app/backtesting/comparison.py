"""Side-by-side strategy comparison."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown, summarize_drawdown
from iqrp.app.backtesting.performance.returns import (
    annualized_return,
    as_returns,
    cagr,
    total_return,
)
from iqrp.app.backtesting.performance.risk_adjusted import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from iqrp.app.backtesting.performance.scorecard import StrategyScorecard
from iqrp.app.backtesting.performance.tail import conditional_value_at_risk
from iqrp.app.backtesting.performance.trade_metrics import turnover

__all__ = [
    "compare_strategies",
    "compare_scorecards",
    "compare_configurations",
    "rank_strategies",
]


def _series_metrics(
    returns: Any,
    *,
    positions: Any | None = None,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
) -> dict[str, float]:
    r = as_returns(returns)
    return {
        "total_return": total_return(r),
        "cagr": cagr(r, periods_per_year=periods_per_year),
        "annualized_return": annualized_return(r, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free=risk_free, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(r, mar=risk_free, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(r, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(r),
        "cvar": conditional_value_at_risk(r),
        "turnover": float(turnover(positions)) if positions is not None else 0.0,
        "n_obs": float(r.size),
    }


def compare_strategies(
    strategies: Mapping[str, Any],
    *,
    positions: Mapping[str, Any] | None = None,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    include_drawdown_detail: bool = False,
) -> dict[str, Any]:
    """Side-by-side comparison of strategy return series.

    ``strategies`` maps name → returns. Optional ``positions`` maps name → weights.
    """
    table: dict[str, dict[str, float]] = {}
    for name, rets in strategies.items():
        pos = None if positions is None else positions.get(name)
        metrics = _series_metrics(
            rets,
            positions=pos,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )
        if include_drawdown_detail:
            metrics.update(
                {f"dd_{k}": float(v) if isinstance(v, (int, float, np.floating)) else np.nan
                 for k, v in summarize_drawdown(rets).items()
                 if isinstance(v, (int, float, np.floating)) or v is None}
            )
        table[str(name)] = metrics

    ranking = rank_strategies(table, metric="sharpe")
    return {
        "name": "strategy_comparison",
        "strategies": table,
        "ranking": ranking,
        "metric_names": sorted({k for m in table.values() for k in m}),
    }


def compare_scorecards(
    scorecards: Mapping[str, StrategyScorecard | Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare standardized :class:`StrategyScorecard` objects."""
    rows: dict[str, dict[str, Any]] = {}
    for name, sc in scorecards.items():
        if isinstance(sc, StrategyScorecard):
            rows[str(name)] = sc.to_dict()
        else:
            rows[str(name)] = dict(sc)
    # Rank by a composite that is not Sharpe-only
    scores = {}
    for name, row in rows.items():
        sharpe = float(row.get("sharpe", 0.0))
        mdd = float(row.get("max_drawdown", 0.0))
        stab = float(row.get("stability", 0.0))
        oos = row.get("out_of_sample")
        oos_v = float(oos) if oos is not None else sharpe
        costs = float(row.get("transaction_costs", 0.0))
        composite = sharpe + 0.5 * oos_v + 0.25 * stab - 2.0 * mdd - 0.5 * costs
        scores[name] = composite
    ranking = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
    return {
        "name": "scorecard_comparison",
        "scorecards": rows,
        "composite_scores": scores,
        "ranking": ranking,
    }


def rank_strategies(
    metrics_table: Mapping[str, Mapping[str, float]],
    *,
    metric: str = "sharpe",
    ascending: bool = False,
) -> list[str]:
    """Rank strategy names by a metric column."""
    items = []
    for name, row in metrics_table.items():
        val = row.get(metric, float("-inf") if not ascending else float("inf"))
        items.append((str(name), float(val)))
    items.sort(key=lambda x: x[1], reverse=not ascending)
    return [n for n, _ in items]


def compare_configurations(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    returns_key: str = "returns",
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Compare versions / parameter sets / models / execution / risk configs.

    Each config mapping must contain a ``returns`` series (or ``returns_key``).
    Extra keys are retained as metadata.
    """
    strategies = {}
    meta = {}
    for name, cfg in configs.items():
        strategies[name] = cfg[returns_key]
        meta[name] = {k: v for k, v in cfg.items() if k != returns_key and not _is_array(v)}
    comp = compare_strategies(strategies, periods_per_year=periods_per_year)
    comp["metadata"] = meta
    return comp


def _is_array(v: Any) -> bool:
    return isinstance(v, (np.ndarray, list, tuple))
