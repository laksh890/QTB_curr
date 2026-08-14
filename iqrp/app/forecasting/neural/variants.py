"""Variant neural models: stacked / bidirectional LSTM-GRU and attention Seq2Seq."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.neural.config import NeuralSettings
from iqrp.app.forecasting.neural.gru.model import GRUForecastModel
from iqrp.app.forecasting.neural.lstm.model import LSTMForecastModel
from iqrp.app.forecasting.neural.seq2seq.model import Seq2SeqForecastModel


def _merge(
    settings: Any | None, patch: dict[str, Any], **params: Any
) -> tuple[NeuralSettings, dict[str, Any]]:
    base: dict[str, Any] = {}
    if isinstance(settings, dict):
        base = dict(settings)
    elif settings is None:
        base = {}
    elif isinstance(settings, NeuralSettings):
        base = settings.model_dump()
    else:
        base = NeuralSettings.default().model_dump()
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return NeuralSettings.from_mapping(base), params


@register_forecast_model
class StackedLSTMForecastModel(LSTMForecastModel):
    architecture_name = "stacked_lstm"
    meta = ForecastModelMeta(
        name="stacked_lstm",
        version="1.0.0",
        description="Stacked LSTM neural forecaster",
        algorithm_family="neural",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        s, params = _merge(
            settings, {"architecture": {"num_layers": 3, "bidirectional": False}}, **params
        )
        super().__init__(settings=s, **params)


@register_forecast_model
class BidirectionalLSTMForecastModel(LSTMForecastModel):
    architecture_name = "bidirectional_lstm"
    meta = ForecastModelMeta(
        name="bidirectional_lstm",
        version="1.0.0",
        description="Bidirectional LSTM neural forecaster",
        algorithm_family="neural",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        s, params = _merge(
            settings, {"architecture": {"bidirectional": True, "num_layers": 2}}, **params
        )
        super().__init__(settings=s, **params)


@register_forecast_model
class StackedGRUForecastModel(GRUForecastModel):
    architecture_name = "stacked_gru"
    meta = ForecastModelMeta(
        name="stacked_gru",
        version="1.0.0",
        description="Stacked GRU neural forecaster",
        algorithm_family="neural",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        s, params = _merge(
            settings, {"architecture": {"num_layers": 3, "bidirectional": False}}, **params
        )
        super().__init__(settings=s, **params)


@register_forecast_model
class AttentionSeq2SeqForecastModel(Seq2SeqForecastModel):
    architecture_name = "attention_seq2seq"
    meta = ForecastModelMeta(
        name="attention_seq2seq",
        version="1.0.0",
        description="Attention-based Encoder-Decoder Seq2Seq forecaster",
        algorithm_family="neural",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        s, params = _merge(settings, {"architecture": {"attention": True}}, **params)
        super().__init__(settings=s, **params)
