"""MLP backbone."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.heads import output_head, reshape_head
from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    nn = object  # type: ignore[assignment]


class MLPNet(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        lookback: int,
        horizon: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        task: str = "regression",
        n_classes: int = 2,
        n_quantiles: int = 3,
        dist: bool = False,
    ) -> None:
        if has_torch():
            super().__init__()
        self.horizon = horizon
        self.task = task
        self.n_classes = n_classes
        self.n_quantiles = n_quantiles
        self.dist = dist
        layers: list[Any] = []
        in_dim = n_features * lookback
        for i in range(max(num_layers, 1)):
            out_dim = hidden_size
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            if i == 0:
                layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.body = nn.Sequential(*layers)
        self.head = output_head(
            hidden_size, horizon, task=task, n_classes=n_classes, n_quantiles=n_quantiles, dist=dist
        )

    def forward(self, x: Any) -> Any:
        # x: (B, T, F)
        b = x.shape[0]
        h = self.body(x.reshape(b, -1))
        out = self.head(h)
        return reshape_head(
            out, b, self.horizon, task=self.task, n_classes=self.n_classes, n_quantiles=self.n_quantiles, dist=self.dist
        )
