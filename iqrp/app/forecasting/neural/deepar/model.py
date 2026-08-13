"""DeepAR forecasting model."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel
from iqrp.app.forecasting.neural.deepar.net import DeepARNet


@register_forecast_model
class DeepARForecastModel(NeuralForecastModel):
    architecture_name = "deepar"
    meta = ForecastModelMeta(
        name="deepar",
        version="1.0.0",
        description="DeepAR probabilistic RNN forecaster",
        algorithm_family="neural",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=False,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        # default to gaussian NLL for DeepAR
        if settings is None:
            from iqrp.app.forecasting.neural.config import NeuralSettings

            settings = NeuralSettings.from_mapping(
                {
                    "train": {"loss": "gaussian_nll"},
                    "probabilistic": {"enabled": True, "distribution": "gaussian"},
                    "task": {"type": "distribution"},
                }
            )
        elif isinstance(settings, dict) and "train" not in settings:
            settings = {**settings, "train": {"loss": "gaussian_nll"}, "task": {"type": "distribution"}}
        super().__init__(settings=settings, **params)

    def _build_module(self, *, n_features: int, task: str) -> Any:
        kw = self._arch_kwargs()
        return DeepARNet(
            n_features,
            self._horizon,
            hidden_size=kw["hidden_size"],
            num_layers=kw["num_layers"],
            dropout=kw["dropout"],
            distribution=self._neural_settings.probabilistic.distribution,
        )
