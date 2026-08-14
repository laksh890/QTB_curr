"""Scaled multi-head attention with optional RoPE."""

from __future__ import annotations

import math
from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.base.masking import apply_mask_to_scores

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class MultiHeadAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        assert d_model % max(n_heads, 1) == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn: Any = None

    def forward(self, query: Any, key: Any, value: Any, mask: Any = None) -> Any:
        b = query.size(0)
        q = self.q(query).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(key).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(value).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = apply_mask_to_scores(scores, mask)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        self.last_attn = attn.detach()
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.out(ctx)
