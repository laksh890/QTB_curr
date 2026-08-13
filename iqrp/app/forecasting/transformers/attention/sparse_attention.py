"""Sparse / local+global attention."""

from __future__ import annotations

import math
from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.base.masking import apply_mask_to_scores, local_attention_mask

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class SparseAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, window: int = 16, n_global: int = 4) -> None:
        if has_torch():
            super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // max(n_heads, 1)
        self.d_model = d_model
        self.window = window
        self.n_global = n_global
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn: Any = None

    def forward(self, query: Any, key: Any, value: Any, mask: Any = None) -> Any:
        b, t, _ = query.shape
        q = self.q(query).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(key).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(value).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        local = local_attention_mask(t, self.window, device=query.device)
        # allow global tokens (first n_global)
        if self.n_global > 0:
            local[: self.n_global, :] = False
            local[:, : self.n_global] = False
        scores = apply_mask_to_scores(scores, local)
        scores = apply_mask_to_scores(scores, mask)
        attn = self.drop(torch.softmax(scores, dim=-1))
        self.last_attn = attn.detach()
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(ctx)
