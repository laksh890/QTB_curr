"""Institutional Feature Validation and Statistical Research Engine.

Downstream models should consume only features that pass validation through
:class:`FeatureResearchValidator`. This package does not generate trading
signals — it produces quantitative evidence about predictive usefulness,
redundancy, stability, and drift.
"""

from iqrp.app.features.research.config import ResearchSettings
from iqrp.app.features.research.correlation import CorrelationAnalyzer, CorrelationReport
from iqrp.app.features.research.drift import DriftDetector
from iqrp.app.features.research.feature_statistics import FeatureStatisticsEngine
from iqrp.app.features.research.importance import ImportanceAnalyzer
from iqrp.app.features.research.predictive_power import PredictivePowerEngine
from iqrp.app.features.research.redundancy import RedundancyDetector
from iqrp.app.features.research.reports import ReportWriter, ResearchReportDocument
from iqrp.app.features.research.stability import StabilityAnalyzer
from iqrp.app.features.research.validator import FeatureResearchResult, FeatureResearchValidator
from iqrp.app.features.research.visualization import ResearchVisualizer

__all__ = [
    "CorrelationAnalyzer",
    "CorrelationReport",
    "DriftDetector",
    "FeatureResearchResult",
    "FeatureResearchValidator",
    "FeatureStatisticsEngine",
    "ImportanceAnalyzer",
    "PredictivePowerEngine",
    "RedundancyDetector",
    "ReportWriter",
    "ResearchReportDocument",
    "ResearchSettings",
    "ResearchVisualizer",
    "StabilityAnalyzer",
]
