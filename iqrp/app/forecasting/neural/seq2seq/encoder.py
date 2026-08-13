"""Seq2Seq encoder."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    nn = object  # type: ignore[assignment]


class Encoder(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        if has_torch():
            super().__init__()
        self.rnn = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: Any) -> tuple[Any, tuple[Any, Any]]:
        outputs, state = self.rnn(x)
        return outputs, state
