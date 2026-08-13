"""Institutional Risk Intelligence Framework.

Central risk gate between Alpha / Forecasting and Portfolio / Execution.
Risk never generates alpha. Hard limits cannot be overridden by forecast confidence.

Integration hook (Phase 09 completion):
- ``CapitalAllocator`` — institutional capital / risk-budget distribution
- ``RiskIntelligenceEnsemble`` — unified multi-dimension risk decision layer

Both consume existing risk engines via import-only contracts; they do not
reimplement VaR / CVaR / portfolio risk / Kelly sizing.
"""

from iqrp.app.risk.base import (
    LimitBreach,
    LimitSeverity,
    RiskDecision,
    RiskLimit,
    RiskMeasure,
    RiskModel,
    RiskReport,
    RiskState,
    as_returns,
    as_weights,
    build_report,
    evaluate_limits,
)
from iqrp.app.risk.capital import CapitalAllocation, CapitalAllocator, CapitalSettings, RiskBudget
from iqrp.app.risk.config import RiskSettings
from iqrp.app.risk.ensemble import (
    DecisionAction,
    EnsembleDecision,
    EnsembleSettings,
    RiskAssessment,
    RiskIntelligenceEnsemble,
    RiskScore,
)
from iqrp.app.risk.orchestrator import RiskIntelligenceEngine

__all__ = [
    "RiskIntelligenceEngine",
    "RiskSettings",
    "CapitalAllocator",
    "CapitalSettings",
    "CapitalAllocation",
    "RiskBudget",
    "RiskIntelligenceEnsemble",
    "EnsembleSettings",
    "RiskScore",
    "RiskAssessment",
    "EnsembleDecision",
    "DecisionAction",
    "RiskModel",
    "RiskMeasure",
    "RiskReport",
    "RiskLimit",
    "RiskDecision",
    "RiskState",
    "LimitSeverity",
    "LimitBreach",
    "evaluate_limits",
    "build_report",
    "as_returns",
    "as_weights",
]
