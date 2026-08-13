"""Institutional Time-Series Transformer Forecasting Platform.

Ultra-long-horizon, probabilistic, multi-asset and regime-aware transformer
forecasting — all models inherit from the Forecasting Framework.
"""

from iqrp.app.forecasting.transformers.config import TransformerSettings
from iqrp.app.forecasting.transformers.registry import (
    create_transformer_model,
    ensure_transformer_models_loaded,
    list_transformer_models,
)
from iqrp.app.forecasting.transformers.trainer import TransformerOrchestrator, TransformerTrainResult

ensure_transformer_models_loaded()

__all__ = [
    "TransformerSettings",
    "TransformerOrchestrator",
    "TransformerTrainResult",
    "create_transformer_model",
    "ensure_transformer_models_loaded",
    "list_transformer_models",
]
