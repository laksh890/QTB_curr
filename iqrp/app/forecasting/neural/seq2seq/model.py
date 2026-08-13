"""Seq2Seq / Attention Seq2Seq forecasting model."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.neural.base.neural_model import NeuralForecastModel
from iqrp.app.forecasting.neural.seq2seq.net import Seq2SeqNet


@register_forecast_model
class Seq2SeqForecastModel(NeuralForecastModel):
    architecture_name = "seq2seq"
    meta = ForecastModelMeta(
        name="seq2seq",
        version="1.0.0",
        description="Encoder-Decoder Seq2Seq with optional attention",
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
        return Seq2SeqNet(
            n_features,
            self._horizon,
            hidden_size=kw["hidden_size"],
            num_layers=max(kw["num_layers"], 1),
            dropout=kw["dropout"],
            use_attention=kw["use_attention"],
            task=task,
            n_quantiles=kw["n_quantiles"],
            dist=kw["dist"],
        )
