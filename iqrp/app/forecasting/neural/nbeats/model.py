"""N-BEATS forecasting model."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel
from iqrp.app.forecasting.neural.nbeats.net import NBeatsNet


@register_forecast_model
class NBeatsForecastModel(NeuralForecastModel):
    architecture_name = "nbeats"
    meta = ForecastModelMeta(
        name="nbeats",
        version="1.0.0",
        description="N-BEATS neural basis expansion forecaster",
        algorithm_family="neural",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=False,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def _build_module(self, *, n_features: int, task: str) -> Any:
        kw = self._arch_kwargs()
        return NBeatsNet(
            n_features,
            self._lookback,
            self._horizon,
            hidden_size=kw["hidden_size"],
            n_blocks=kw["n_blocks"],
            task=task,
            n_quantiles=kw["n_quantiles"],
            dist=kw["dist"],
        )
