"""Backtest performance metrics package."""

from iqrp.app.backtesting.performance.attribution import full_attribution
from iqrp.app.backtesting.performance.benchmark import compare_to_benchmark
from iqrp.app.backtesting.performance.drawdown import (
    max_drawdown,
    summarize_drawdown,
)
from iqrp.app.backtesting.performance.exposure import summarize_exposure
from iqrp.app.backtesting.performance.returns import (
    annualized_return,
    cagr,
    summarize_returns,
    total_return,
)
from iqrp.app.backtesting.performance.risk_adjusted import (
    sharpe_ratio,
    sortino_ratio,
    summarize_risk_adjusted,
)
from iqrp.app.backtesting.performance.scorecard import StrategyScorecard, build_scorecard
from iqrp.app.backtesting.performance.stability import stability_report
from iqrp.app.backtesting.performance.tail import summarize_tail
from iqrp.app.backtesting.performance.trade_metrics import summarize_trades

__all__ = [
    "total_return",
    "cagr",
    "annualized_return",
    "summarize_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize_risk_adjusted",
    "max_drawdown",
    "summarize_drawdown",
    "summarize_tail",
    "summarize_trades",
    "summarize_exposure",
    "full_attribution",
    "compare_to_benchmark",
    "stability_report",
    "StrategyScorecard",
    "build_scorecard",
]
