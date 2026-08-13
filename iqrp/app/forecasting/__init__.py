"""Institutional Forecasting Framework.

Common infrastructure for every forecasting algorithm in IQRP.
Concrete algorithms (ARIMA, LSTM, …) plug in via ``@register_forecast_model``.
"""

from iqrp.app.forecasting.base import (
    EvaluationReport,
    Forecast,
    ForecastEvaluator,
    ForecastModel,
    ForecastModelMeta,
    ForecastTrainer,
    Prediction,
    PredictionInterval,
    TrainingMetadata,
    ensure_forecast_models_loaded,
    get_registry,
    register_forecast_model,
)
from iqrp.app.forecasting.config import ForecastingSettings
from iqrp.app.forecasting.orchestration import ForecastingPipeline, ForecastScheduler
from iqrp.app.forecasting.serialization import ForecastSerializer

# Eager-load framework stub so the registry is non-empty on import.
ensure_forecast_models_loaded()

__all__ = [
    "EvaluationReport",
    "Forecast",
    "ForecastEvaluator",
    "ForecastModel",
    "ForecastModelMeta",
    "ForecastScheduler",
    "ForecastSerializer",
    "ForecastTrainer",
    "ForecastingPipeline",
    "ForecastingSettings",
    "Prediction",
    "PredictionInterval",
    "TrainingMetadata",
    "ensure_forecast_models_loaded",
    "get_registry",
    "register_forecast_model",
]
