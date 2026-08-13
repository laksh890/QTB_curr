"""FlashAttention-style memory-efficient attention (chunked SDPA fallback)."""

from __future__ import annotations

import math
from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class FlashAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    """Uses torch SDPA when available; otherwise chunked attention."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0, chunk_size: int = 256) -> None:
        if has_torch():
            super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // max(n_heads, 1)
        self.d_model = d_model
        self.chunk_size = max(int(chunk_size), 16)
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
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )
            self.last_attn = None
        else:  # pragma: no cover
            out = self._chunked(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(self.drop(out))

    def _chunked(self, q: Any, k: Any, v: Any) -> Any:
        b, h, t, d = q.shape
        outs = []
        for start in range(0, t, self.chunk_size):
            q_c = q[:, :, start : start + self.chunk_size]
            scores = torch.matmul(q_c, k.transpose(-2, -1)) / math.sqrt(d)
            attn = torch.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, v))
        return torch.cat(outs, dim=2)
