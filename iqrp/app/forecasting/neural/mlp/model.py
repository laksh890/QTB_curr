"""MLP forecasting model."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel
from iqrp.app.forecasting.neural.mlp.net import MLPNet


@register_forecast_model
class MLPForecastModel(NeuralForecastModel):
    architecture_name = "mlp"
    meta = ForecastModelMeta(
        name="mlp",
        version="1.0.0",
        description="Multi-Layer Perceptron neural forecaster",
        algorithm_family="neural",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def _build_module(self, *, n_features: int, task: str) -> Any:
        kw = self._arch_kwargs()
        return MLPNet(
            n_features,
            self._lookback,
            self._horizon,
            hidden_size=kw["hidden_size"],
            num_layers=kw["num_layers"],
            dropout=kw["dropout"],
            task=task,
            n_classes=kw["n_classes"],
            n_quantiles=kw["n_quantiles"],
            dist=kw["dist"],
        )
