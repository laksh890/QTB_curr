"""Institutional Forecast Intelligence Platform.

The sole forecasting interface consumed by Risk, Portfolio, Execution,
Trading Bot, and Research. Discovers, benchmarks, ranks, tunes, ensembles,
calibrates, monitors, and deploys every registered forecast model.
"""

from iqrp.app.forecasting.intelligence.config import IntelligenceSettings
from iqrp.app.forecasting.intelligence.orchestrator import ForecastIntelligenceEngine
from iqrp.app.forecasting.intelligence.registry import (
    create_model,
    discover_engine_modules,
    list_discovered_models,
    load_discovered_engines,
)

__all__ = [
    "ForecastIntelligenceEngine",
    "IntelligenceSettings",
    "create_model",
    "discover_engine_modules",
    "list_discovered_models",
    "load_discovered_engines",
]
