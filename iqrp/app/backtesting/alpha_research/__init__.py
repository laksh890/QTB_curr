"""Alpha research & signal discovery (backtesting-integrated research platform).

Separates FEATURE → SIGNAL → POSITION → (risk/execution when cascaded) → PERFORMANCE.
Not a profitability engine. Short development samples are marked SAMPLE TOO SHORT.
"""

from iqrp.app.backtesting.alpha_research.engine import AlphaSignalResearchEngine
from iqrp.app.backtesting.alpha_research.experiments import ExperimentRegistry, ExperimentSpec
from iqrp.app.backtesting.alpha_research.features import FeatureRegistry, FeatureSpec, get_feature_registry
from iqrp.app.backtesting.alpha_research.leakage import LeakageError, run_leakage_suite
from iqrp.app.backtesting.alpha_research.signals import SignalRegistry, SignalSpec, get_signal_registry
from iqrp.app.backtesting.alpha_research.types import (
    AlphaClassification,
    SAMPLE_TOO_SHORT_DISCLAIMER,
    TimeframeContext,
)

__all__ = [
    "AlphaClassification",
    "AlphaSignalResearchEngine",
    "ExperimentRegistry",
    "ExperimentSpec",
    "FeatureRegistry",
    "FeatureSpec",
    "LeakageError",
    "SAMPLE_TOO_SHORT_DISCLAIMER",
    "SignalRegistry",
    "SignalSpec",
    "TimeframeContext",
    "get_feature_registry",
    "get_signal_registry",
    "run_leakage_suite",
]
