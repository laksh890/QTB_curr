"""Institutional Neural Forecasting Platform.

Short-term, long-term, multi-horizon, probabilistic, classification,
regression and regime-aware sequence forecasting — all inheriting from
the Forecasting Framework.
"""

from iqrp.app.forecasting.neural.config import NeuralSettings
from iqrp.app.forecasting.neural.registry import (
    create_neural_model,
    ensure_neural_models_loaded,
    list_neural_models,
)
from iqrp.app.forecasting.neural.trainer import NeuralOrchestrator, NeuralTrainResult

ensure_neural_models_loaded()

__all__ = [
    "NeuralSettings",
    "NeuralOrchestrator",
    "NeuralTrainResult",
    "create_neural_model",
    "ensure_neural_models_loaded",
    "list_neural_models",
]
