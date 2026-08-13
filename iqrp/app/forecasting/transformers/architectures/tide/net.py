"""TiDE network."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.base.heads import forecast_head, reshape_forecast

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class TiDENet(nn.Module if has_torch() else object):  # type: ignore[misc]
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
        self.proj = nn.Linear(n_features, d_model)
        enc: list[Any] = []
        for _ in range(max(num_layers, 1)):
            enc += [
                nn.Linear(d_model, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, d_model),
                nn.LayerNorm(d_model),
            ]
        self.encoder = nn.Sequential(*enc)
        self.head = forecast_head(
            d_model, horizon, task=task, n_classes=n_classes, n_quantiles=n_quantiles, dist=dist
        )

    def encode(self, x: Any) -> Any:
        # dense temporal encoding via mean-pooled projected features
        h = self.proj(x).mean(dim=1)
        return self.encoder(h).unsqueeze(1)

    def forward(self, x: Any) -> Any:
        h = self.encode(x).squeeze(1)
        out = self.head(h)
        return reshape_forecast(
            out,
            x.shape[0],
            self.horizon,
            task=self.task,
            n_classes=self.n_classes,
            n_quantiles=self.n_quantiles,
            dist=self.dist,
        )
