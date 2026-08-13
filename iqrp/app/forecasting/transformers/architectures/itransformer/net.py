"""iTransformer network."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.base.encoder import TransformerEncoder
from iqrp.app.forecasting.transformers.base.heads import forecast_head, reshape_forecast

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class iTransformerNet(nn.Module if has_torch() else object):  # type: ignore[misc]
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
        # project each variate's time series via 1d conv over time (length-agnostic)
        self.var_proj = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.encoder = TransformerEncoder(d_model, n_heads, num_layers, ffn_dim, dropout, attention_type)
        self.head = forecast_head(
            d_model, horizon, task=task, n_classes=n_classes, n_quantiles=n_quantiles, dist=dist
        )

    def encode(self, x: Any) -> Any:
        # x (B,T,F) -> tokens (B,F,D)
        b, t, f = x.shape
        tokens = []
        for i in range(f):
            series = x[:, :, i : i + 1].transpose(1, 2)  # (B,1,T)
            tokens.append(self.var_proj(series).squeeze(-1))  # (B,D)
        h = torch.stack(tokens, dim=1)
        return self.encoder(h)

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
