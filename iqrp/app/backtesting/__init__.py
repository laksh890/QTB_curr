"""Institutional backtesting platform."""

from iqrp.app.backtesting.capacity import capacity_curve, estimate_capacity_limit
from iqrp.app.backtesting.comparison import compare_strategies
from iqrp.app.backtesting.config import BacktestSettings
from iqrp.app.backtesting.engine import BacktestEngine, BacktestResult
from iqrp.app.backtesting.experiment_registry import ExperimentLineage, ExperimentRegistry
from iqrp.app.backtesting.paper_trading import PaperTradingConfig, PaperTradingInterface
from iqrp.app.backtesting.performance import StrategyScorecard, build_scorecard, sharpe_ratio
from iqrp.app.backtesting.robustness import ablation_test, parameter_sweep
from iqrp.app.backtesting.runner import BacktestRunConfig, BacktestRunner, OperationalBacktestResult
from iqrp.app.backtesting.scenarios import HistoricalScenario, ScenarioEngine
from iqrp.app.backtesting.types import BacktestState
from iqrp.app.backtesting.validation_gates import GateResult, GateThresholds, evaluate_gates

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestRunConfig",
    "BacktestRunner",
    "BacktestSettings",
    "BacktestState",
    "ExperimentLineage",
    "ExperimentRegistry",
    "GateResult",
    "GateThresholds",
    "OperationalBacktestResult",
    "PaperTradingConfig",
    "PaperTradingInterface",
    "StrategyScorecard",
    "build_scorecard",
    "evaluate_gates",
    "sharpe_ratio",
    "ScenarioEngine",
    "HistoricalScenario",
    "capacity_curve",
    "estimate_capacity_limit",
    "compare_strategies",
    "parameter_sweep",
    "ablation_test",
]
