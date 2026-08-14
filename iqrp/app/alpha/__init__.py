"""Institutional Alpha Research foundation.

CRITICAL RULES:
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Must track economic_hypothesis on SignalDefinition.
- Point-in-time: no future leakage in signal computation helpers (past windows only).
- Alpha approval ≠ trading approval (Risk Intelligence is not bypassed).

Discovery templates produce research candidates without claiming profitability.
"""

from iqrp.app.alpha.base import (
    AlphaSignal,
    ExperimentRecord,
    SignalDefinition,
    SignalMetadata,
    SignalPerformance,
    SignalRegistry,
    SignalResearchReport,
    SignalScore,
    SignalStatistics,
    SignalStatus,
    StatusTransition,
    get_default_registry,
    validate_transition,
)
from iqrp.app.alpha.config import AlphaSettings
from iqrp.app.alpha.engine import AlphaResearchEngine, ApprovalError
from iqrp.app.alpha.phase11 import validate_phase11, write_phase11_report
from iqrp.app.alpha.registry import (
    available as registry_available,
    get as registry_get,
    register as registry_register,
)
from iqrp.app.alpha.serializer import AlphaSerializer

__all__ = [
    "AlphaResearchEngine",
    "AlphaSerializer",
    "AlphaSettings",
    "AlphaSignal",
    "ApprovalError",
    "ExperimentRecord",
    "SignalDefinition",
    "SignalMetadata",
    "SignalPerformance",
    "SignalRegistry",
    "SignalResearchReport",
    "SignalScore",
    "SignalStatistics",
    "SignalStatus",
    "StatusTransition",
    "get_default_registry",
    "registry_available",
    "registry_get",
    "registry_register",
    "validate_phase11",
    "validate_transition",
    "write_phase11_report",
]
