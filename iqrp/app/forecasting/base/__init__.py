"""Forecasting framework base contracts."""

from iqrp.app.forecasting.base.evaluator import EvaluationReport, ForecastEvaluator
from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.metadata import ForecastModelMeta, TrainingMetadata
from iqrp.app.forecasting.base.prediction import (
    DistributionForecast,
    Prediction,
    PredictionInterval,
    QuantileForecast,
)
from iqrp.app.forecasting.base.registry import (
    ensure_forecast_models_loaded,
    forecast_model_factory,
    get_registry,
    register_forecast_model,
)
from iqrp.app.forecasting.base.trainer import ForecastTrainer, TrainResult

__all__ = [
    "DistributionForecast",
    "EvaluationReport",
    "Forecast",
    "ForecastEvaluator",
    "ForecastModel",
    "ForecastModelMeta",
    "ForecastTrainer",
    "Prediction",
    "PredictionInterval",
    "QuantileForecast",
    "TrainResult",
    "TrainingMetadata",
    "ensure_forecast_models_loaded",
    "forecast_model_factory",
    "get_registry",
    "register_forecast_model",
]
