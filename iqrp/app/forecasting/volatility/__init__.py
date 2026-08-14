"""Institutional Volatility Forecasting Engine.

All volatility forecasts for Risk, Portfolio, Execution, and Trading Bot
must be consumed exclusively through this package.
"""

from iqrp.app.forecasting.volatility.config import VolatilitySettings
from iqrp.app.forecasting.volatility.registry import (
    create_volatility_model,
    ensure_volatility_models_loaded,
    list_volatility_models,
)
from iqrp.app.forecasting.volatility.trainer import VolatilityTrainer, VolatilityTrainResult

ensure_volatility_models_loaded()

__all__ = [
    "VolatilitySettings",
    "VolatilityTrainResult",
    "VolatilityTrainer",
    "create_volatility_model",
    "ensure_volatility_models_loaded",
    "list_volatility_models",
]
