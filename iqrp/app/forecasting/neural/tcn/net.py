"""Temporal Convolutional Network backbone."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.heads import output_head, reshape_head
from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class TemporalBlock(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float) -> None:
        if has_torch():
            super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.down = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        self.padding = padding

    def forward(self, x: Any) -> Any:
        y = self.conv1(x)
        if self.padding > 0:
            y = y[..., : -self.padding]
        y = self.dropout(self.relu(self.norm1(y)))
        y2 = self.conv2(y)
        if self.padding > 0:
            y2 = y2[..., : -self.padding]
        y2 = self.dropout(self.relu(self.norm2(y2)))
        res = self.down(x)
        # align lengths
        t = min(y2.shape[-1], res.shape[-1])
        return self.relu(y2[..., :t] + res[..., :t])


class TCNNet(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        horizon: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
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
        layers = []
        for i in range(max(num_layers, 1)):
            din = n_features if i == 0 else hidden_size
            layers.append(TemporalBlock(din, hidden_size, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.head = output_head(
            hidden_size, horizon, task=task, n_classes=n_classes, n_quantiles=n_quantiles, dist=dist
        )

    def forward(self, x: Any) -> Any:
        # x: (B,T,F) -> (B,F,T)
        b = x.shape[0]
        h = self.network(x.transpose(1, 2))
        last = h[:, :, -1]
        pred = self.head(last)
        return reshape_head(
            pred, b, self.horizon, task=self.task, n_classes=self.n_classes, n_quantiles=self.n_quantiles, dist=self.dist
        )
