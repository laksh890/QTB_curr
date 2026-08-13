"""Institutional Capital Allocation Engine."""

from iqrp.app.risk.capital.allocator import CapitalAllocator
from iqrp.app.risk.capital.capital_budget import allocate_capital_budgets
from iqrp.app.risk.capital.capacity import estimate_capacity
from iqrp.app.risk.capital.config import CapitalSettings
from iqrp.app.risk.capital.correlation import (
    correlation_crowding_scales,
    effective_risk_budgets,
    strategy_correlation,
    tail_dependence_matrix,
)
from iqrp.app.risk.capital.diagnostics import diagnose_allocation, diagnose_covariance, diagnose_weights
from iqrp.app.risk.capital.drawdown import drawdown_scales
from iqrp.app.risk.capital.dynamic import dynamic_risk_scales
from iqrp.app.risk.capital.equal_risk import equal_risk_weights
from iqrp.app.risk.capital.evaluator import evaluate_allocation
from iqrp.app.risk.capital.hierarchical import herc_weights, hrp_weights
from iqrp.app.risk.capital.optimizer import optimize_risk_budgets
from iqrp.app.risk.capital.processes import all_capital_scenarios, simulate_capital_scenario
from iqrp.app.risk.capital.risk_budget import build_risk_budgets
from iqrp.app.risk.capital.risk_parity import capital_risk_parity
from iqrp.app.risk.capital.serializer import CapitalSerializer
from iqrp.app.risk.capital.types import CapitalAllocation, RiskBudget, StrategyAllocation
from iqrp.app.risk.capital.volatility import volatility_budgets

__all__ = [
    "CapitalAllocator",
    "CapitalSettings",
    "CapitalAllocation",
    "RiskBudget",
    "StrategyAllocation",
    "CapitalSerializer",
    "allocate_capital_budgets",
    "build_risk_budgets",
    "capital_risk_parity",
    "equal_risk_weights",
    "hrp_weights",
    "herc_weights",
    "volatility_budgets",
    "estimate_capacity",
    "strategy_correlation",
    "tail_dependence_matrix",
    "correlation_crowding_scales",
    "effective_risk_budgets",
    "drawdown_scales",
    "dynamic_risk_scales",
    "evaluate_allocation",
    "diagnose_allocation",
    "diagnose_covariance",
    "diagnose_weights",
    "optimize_risk_budgets",
    "simulate_capital_scenario",
    "all_capital_scenarios",
]
