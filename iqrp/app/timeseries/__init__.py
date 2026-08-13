"""Institutional Time-Series Analytics Platform.

Reusable analytical primitives for discovering structure in financial
time series. This is NOT a forecasting engine and does NOT emit trading
signals — only measurements and statistical evidence.
"""

from iqrp.app.timeseries.config import TimeSeriesSettings
from iqrp.app.timeseries.orchestrator import TimeSeriesAnalyticsEngine
from iqrp.app.timeseries.registry import ensure_timeseries_loaded, list_methods
from iqrp.app.timeseries.multiple_testing import adjust_pvalues

ensure_timeseries_loaded()

__all__ = [
    "TimeSeriesSettings",
    "TimeSeriesAnalyticsEngine",
    "ensure_timeseries_loaded",
    "list_methods",
    "adjust_pvalues",
]
