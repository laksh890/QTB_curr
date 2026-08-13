"""Temporal Convolutional Network forecasting model."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel
from iqrp.app.forecasting.neural.tcn.net import TCNNet


@register_forecast_model
class TCNForecastModel(NeuralForecastModel):
    architecture_name = "tcn"
    meta = ForecastModelMeta(
        name="tcn",
        version="1.0.0",
        description="Temporal Convolutional Network forecaster",
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
        return TCNNet(
            n_features,
            self._horizon,
            hidden_size=kw["hidden_size"],
            num_layers=kw["num_layers"],
            kernel_size=kw["kernel_size"],
            dropout=kw["dropout"],
            task=task,
            n_classes=kw["n_classes"],
            n_quantiles=kw["n_quantiles"],
            dist=kw["dist"],
        )
