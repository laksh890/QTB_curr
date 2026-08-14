"""Temporal Fusion Transformer network."""

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


class GatedResidualNetwork(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Any) -> Any:
        h = torch.nn.functional.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + h * g)


class TFTNet(nn.Module if has_torch() else object):  # type: ignore[misc]
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
        self.input_emb = TransformerInputEmbedding(
            n_features, d_model, n_regimes=n_regimes, use_regime=use_regime, dropout=dropout
        )
        self.vsn = GatedResidualNetwork(d_model, dropout)
        self.pos = build_positional("sinusoidal", d_model)
        self.encoder = TransformerEncoder(
            d_model, n_heads, num_layers, ffn_dim, dropout, attention_type
        )
        from iqrp.app.forecasting.transformers.base.decoder import TransformerDecoder

        self.decoder = TransformerDecoder(d_model, n_heads, 1, ffn_dim, dropout)
        self.head = forecast_head(
            d_model, horizon, task=task, n_classes=n_classes, n_quantiles=n_quantiles, dist=dist
        )
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)

    def encode(self, x: Any) -> Any:
        h = self.vsn(self.pos(self.input_emb(x)))
        return self.encoder(h)

    def forward(self, x: Any) -> Any:
        mem = self.encode(x)
        q = self.query.expand(x.shape[0], -1, -1)
        h = self.decoder(q, mem)
        out = self.head(h.mean(dim=1))
        return reshape_forecast(
            out,
            x.shape[0],
            self.horizon,
            task=self.task,
            n_classes=self.n_classes,
            n_quantiles=self.n_quantiles,
            dist=self.dist,
        )
