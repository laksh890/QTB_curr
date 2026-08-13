"""Institutional Statistical Forecasting Engine.

Classical econometric models inheriting from the Forecasting Framework.
"""

from iqrp.app.forecasting.statistical.config import StatisticalSettings
from iqrp.app.forecasting.statistical.registry import (
    create_statistical_model,
    ensure_statistical_models_loaded,
    list_statistical_models,
)
from iqrp.app.forecasting.statistical.trainer import StatisticalTrainer, StatisticalTrainResult

ensure_statistical_models_loaded()

__all__ = [
    "StatisticalSettings",
    "StatisticalTrainer",
    "StatisticalTrainResult",
    "create_statistical_model",
    "ensure_statistical_models_loaded",
    "list_statistical_models",
]
