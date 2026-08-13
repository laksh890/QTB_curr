"""Temporal calendar-style embeddings."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class TemporalEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, dim: int = 16) -> None:
        if has_torch():
            super().__init__()
            self.proj = nn.Linear(4, dim)

    def forward(self, x: Any) -> Any:
        if x.shape[-1] != 4:
            pad = torch.zeros(*x.shape[:-1], 4, device=x.device, dtype=x.dtype)
            n = min(x.shape[-1], 4)
            pad[..., :n] = x[..., :n]
            x = pad
        return self.proj(x)
