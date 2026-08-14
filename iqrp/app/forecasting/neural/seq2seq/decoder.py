"""Seq2Seq decoder."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class Decoder(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        if has_torch():
            super().__init__()
        self.rnn = nn.LSTM(
            1,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, y_prev: Any, state: Any) -> tuple[Any, Any]:
        out, state = self.rnn(y_prev, state)
        return self.proj(out), state
