"""Positional encodings for sequence models."""

from __future__ import annotations

import math
from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class PositionalEncoding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        if has_torch():
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max(d_model, 1))
            )
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Any) -> Any:
        return x + self.pe[:, : x.size(1)]
