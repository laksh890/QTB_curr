"""Operational backtest runner package."""

from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.lifecycle import RunnerLifecycleState
from iqrp.app.backtesting.runner.result import OperationalBacktestResult
from iqrp.app.backtesting.runner.runner import BacktestRunner

__all__ = [
    "BacktestRunConfig",
    "BacktestRunner",
    "OperationalBacktestResult",
    "RunnerLifecycleState",
]
