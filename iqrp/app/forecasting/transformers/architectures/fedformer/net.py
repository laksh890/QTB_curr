"""FEDformer network."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.base.embeddings import TransformerInputEmbedding
from iqrp.app.forecasting.transformers.base.encoder import TransformerEncoder
from iqrp.app.forecasting.transformers.base.heads import forecast_head, reshape_forecast
from iqrp.app.forecasting.transformers.base.positional_encoding import build_positional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class SeriesDecomp(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, kernel: int = 25) -> None:
        if has_torch():
            super().__init__()
        k = max(int(kernel), 3)
        if k % 2 == 0:
            k += 1
        self.avg = nn.AvgPool1d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Any) -> tuple[Any, Any]:
        # x (B,T,D)
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class FEDformerNet(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        horizon: int,
        *,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        attention_type: str = "full",
        positional: str = "sinusoidal",
        task: str = "regression",
        n_classes: int = 2,
        n_quantiles: int = 3,
        dist: bool = False,
        n_regimes: int = 4,
        use_regime: bool = True,
        patch_len: int = 8,
        stride: int = 4,
        factor: int = 3,
        moving_avg: int = 25,
        **kwargs: Any,
    ) -> None:
        if has_torch():
            super().__init__()
        self.horizon = horizon
        self.task = task
        self.n_classes = n_classes
        self.n_quantiles = n_quantiles
        self.dist = dist
        self.decomp = SeriesDecomp(moving_avg)
        self.input_emb = TransformerInputEmbedding(
            n_features, d_model, n_regimes=n_regimes, use_regime=use_regime, dropout=dropout
        )
        self.pos = build_positional("sinusoidal", d_model)
        self.encoder = TransformerEncoder(
            d_model, n_heads, num_layers, ffn_dim, dropout, attention_type="linear"
        )
        self.trend_proj = nn.Linear(n_features, horizon)
        self.head = forecast_head(
            d_model, horizon, task=task, n_classes=n_classes, n_quantiles=n_quantiles, dist=dist
        )

    def encode(self, x: Any) -> Any:
        seasonal, trend = self.decomp(x)
        h = self.pos(self.input_emb(seasonal))
        return self.encoder(h), trend

    def forward(self, x: Any) -> Any:
        h, trend = self.encode(x)
        seasonal_out = self.head(h.mean(dim=1))
        trend_out = self.trend_proj(trend.mean(dim=1))
        seasonal_r = reshape_forecast(
            seasonal_out,
            x.shape[0],
            self.horizon,
            task=self.task,
            n_classes=self.n_classes,
            n_quantiles=self.n_quantiles,
            dist=self.dist,
        )
        if self.task in {"regression", "sequence"} and seasonal_r.dim() == 2:
            return seasonal_r + trend_out
        return seasonal_r
