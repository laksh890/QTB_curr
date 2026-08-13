"""GRU / Stacked GRU backbone."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.heads import output_head, reshape_head
from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    nn = object  # type: ignore[assignment]


class GRUNet(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        horizon: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
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
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.rnn = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction = 2 if bidirectional else 1
        self.norm = nn.LayerNorm(hidden_size * direction)
        self.drop = nn.Dropout(dropout)
        self.head = output_head(
            hidden_size * direction,
            horizon,
            task=task,
            n_classes=n_classes,
            n_quantiles=n_quantiles,
            dist=dist,
        )

    def forward(self, x: Any) -> Any:
        b = x.shape[0]
        h = self.input_proj(x)
        out, _ = self.rnn(h)
        last = self.drop(self.norm(out[:, -1, :]))
        pred = self.head(last)
        return reshape_head(
            pred, b, self.horizon, task=self.task, n_classes=self.n_classes, n_quantiles=self.n_quantiles, dist=self.dist
        )
