"""Bahdanau / Luong-style attention."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class Attention(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, hidden_size: int) -> None:
        if has_torch():
            super().__init__()
        self.score = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: Any, encoder_outputs: Any) -> tuple[Any, Any]:
        # query: (B, H), encoder_outputs: (B, T, H)
        b, t, h = encoder_outputs.shape
        q = query.unsqueeze(1).expand(-1, t, -1)
        energy = torch.tanh(self.score(torch.cat([q, encoder_outputs], dim=-1)))
        weights = F.softmax(self.v(energy).squeeze(-1), dim=-1)  # (B,T)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights
