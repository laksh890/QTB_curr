"""Temporal attention and hierarchical temporal attention."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.attention.multihead import MultiHeadAttention

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class TemporalAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Any, mask: Any = None) -> Any:
        return self.norm(x + self.attn(x, x, x, mask=mask))


class HierarchicalAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    """Coarse then fine temporal attention via average pooling."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, pool: int = 4) -> None:
        if has_torch():
            super().__init__()
        self.pool = max(int(pool), 1)
        self.coarse = MultiHeadAttention(d_model, n_heads, dropout)
        self.fine = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Any, mask: Any = None) -> Any:
        b, t, d = x.shape
        p = self.pool
        pad = (p - t % p) % p
        if pad:
            x_p = torch.nn.functional.pad(x, (0, 0, 0, pad))
        else:
            x_p = x
        tp = x_p.size(1) // p
        coarse = x_p.view(b, tp, p, d).mean(dim=2)
        coarse = self.coarse(coarse, coarse, coarse)
        # upsample by repeat
        up = coarse.unsqueeze(2).expand(-1, -1, p, -1).reshape(b, tp * p, d)[:, :t]
        fine = self.fine(x + up, x + up, x + up, mask=mask)
        return self.norm(x + fine)
