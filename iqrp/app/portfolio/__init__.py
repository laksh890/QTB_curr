"""Institutional Portfolio Construction.

Canonical entry points for Forecast → Risk → Portfolio consumers.
Does **not** generate alpha — only expresses provided forecasts/signals under constraints.
"""

from iqrp.app.portfolio.base import (
    OptimizationFailureError,
    OptimizationResult,
    Portfolio,
    PortfolioOptimizer,
    PortfolioType,
    Position,
)
from iqrp.app.portfolio.config import (
    ConstraintsConfig,
    CovarianceConfig,
    ExpectedReturnsConfig,
    ObjectiveConfig,
    PortfolioSettings,
)
from iqrp.app.portfolio.construction import (
    PortfolioConstructor,
    PortfolioResult,
    RebalanceBands,
    RebalancePlan,
    RebalanceTrigger,
    TargetPositions,
    TargetWeights,
    build_target_weights,
    plan_rebalance,
    signals_to_raw_weights,
)
from iqrp.app.portfolio.engine import PortfolioConstructionEngine, ValidationReport
from iqrp.app.portfolio.phase10 import validate_phase10, write_phase10_report

__all__ = [
    "ConstraintsConfig",
    "CovarianceConfig",
    "ExpectedReturnsConfig",
    "ObjectiveConfig",
    "OptimizationFailureError",
    "OptimizationResult",
    "Portfolio",
    "PortfolioConstructionEngine",
    "PortfolioConstructor",
    "PortfolioOptimizer",
    "PortfolioResult",
    "PortfolioSettings",
    "PortfolioType",
    "Position",
    "RebalanceBands",
    "RebalancePlan",
    "RebalanceTrigger",
    "TargetPositions",
    "TargetWeights",
    "ValidationReport",
    "build_target_weights",
    "plan_rebalance",
    "signals_to_raw_weights",
    "validate_phase10",
    "write_phase10_report",
]
