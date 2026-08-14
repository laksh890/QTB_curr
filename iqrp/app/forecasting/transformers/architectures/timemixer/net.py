"""TimeMixer network."""

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


class TimeMixerNet(nn.Module if has_torch() else object):  # type: ignore[misc]
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
        self.pos = build_positional("sinusoidal", d_model)
        self.temporal = TransformerEncoder(
            d_model, n_heads, num_layers, ffn_dim, dropout, "temporal"
        )
        self.channel = TransformerEncoder(d_model, n_heads, 1, ffn_dim, dropout, attention_type)
        self.mix = nn.Linear(d_model * 2, d_model)
        self.head = forecast_head(
            d_model, horizon, task=task, n_classes=n_classes, n_quantiles=n_quantiles, dist=dist
        )

    def encode(self, x: Any) -> Any:
        h = self.pos(self.input_emb(x))
        ht = self.temporal(h)
        # channel mixing via transpose features proxy: mean-pool temporal then expand
        hc = self.channel(ht)
        return self.mix(torch.cat([ht, hc], dim=-1))

    def forward(self, x: Any) -> Any:
        h = self.encode(x)
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
