"""Risk Intelligence Ensemble package.

Unified multi-dimension risk gate. Does not reimplement VaR/CVaR — imports
``iqrp.app.risk.*`` and optionally ``RiskIntelligenceEngine``.
"""

from iqrp.app.risk.ensemble.config import EnsembleSettings
from iqrp.app.risk.ensemble.risk_ensemble import RiskIntelligenceEnsemble
from iqrp.app.risk.ensemble.types import (
    DecisionAction,
    EnsembleDecision,
    NormalizedMetric,
    RiskAssessment,
    RiskScore,
)

__all__ = [
    "DecisionAction",
    "EnsembleDecision",
    "EnsembleSettings",
    "NormalizedMetric",
    "RiskAssessment",
    "RiskIntelligenceEnsemble",
    "RiskScore",
]
