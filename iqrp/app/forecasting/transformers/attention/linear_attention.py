"""Linear (kernel) attention."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class LinearAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // max(n_heads, 1)
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_attn: Any = None

    def forward(self, query: Any, key: Any, value: Any, mask: Any = None) -> Any:
        b, t, _ = query.shape
        q = (
            torch.nn.functional.elu(
                self.q(query).view(b, t, self.n_heads, self.d_k).transpose(1, 2)
            )
            + 1
        )
        k = (
            torch.nn.functional.elu(self.k(key).view(b, -1, self.n_heads, self.d_k).transpose(1, 2))
            + 1
        )
        v = self.v(value).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        ctx = torch.einsum("bhnd,bhde->bhne", q, kv) * z.unsqueeze(-1)
        self.last_attn = None
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(self.drop(ctx))
