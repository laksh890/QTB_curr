"""Crossformer forecasting model."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.transformers.architectures.crossformer.net import CrossformerNet
from iqrp.app.forecasting.transformers.base.transformer_model import TransformerForecastModel


@register_forecast_model
class CrossformerForecastModel(TransformerForecastModel):
    architecture_name = "crossformer"
    meta = ForecastModelMeta(
        name="crossformer",
        version="1.0.0",
        description="Crossformer",
        algorithm_family="transformer",
        task="regression",
        default_horizon=8,
        supports_online=True,
        supports_proba=True,
        supports_intervals=True,
        supports_quantiles=True,
    )

    def _build_module(self, *, n_features: int, task: str) -> Any:
        kw = self._arch_kwargs()
        return CrossformerNet(
            n_features,
            self._horizon,
            d_model=kw["d_model"],
            n_heads=kw["n_heads"],
            num_layers=kw["num_layers"],
            ffn_dim=kw["ffn_dim"],
            dropout=kw["dropout"],
            attention_type=kw["attention_type"],
            positional=kw["positional"],
            task=task,
            n_classes=kw["n_classes"],
            n_quantiles=kw["n_quantiles"],
            dist=kw["dist"],
            n_regimes=kw["n_regimes"],
            use_regime=kw["use_regime"],
            patch_len=kw["patch_len"],
            stride=kw["stride"],
            factor=kw["factor"],
            moving_avg=kw["moving_avg"],
        )
