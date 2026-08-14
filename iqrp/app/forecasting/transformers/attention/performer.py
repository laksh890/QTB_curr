"""Performer (FAVOR+) style linear attention approximation."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class PerformerAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self, d_model: int, n_heads: int = 4, dropout: float = 0.1, n_features: int = 64
    ) -> None:
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
        self.register_buffer(
            "proj",
            torch.randn(self.d_k, n_features) / max(self.d_k**0.5, 1e-6),
            persistent=False,
        )
        self.last_attn: Any = None

    def _phi(self, x: Any) -> Any:
        return torch.nn.functional.relu(x @ self.proj) + 1e-6

    def forward(self, query: Any, key: Any, value: Any, mask: Any = None) -> Any:
        b, t, _ = query.shape
        q = self._phi(self.q(query).view(b, t, self.n_heads, self.d_k).transpose(1, 2))
        k = self._phi(self.k(key).view(b, -1, self.n_heads, self.d_k).transpose(1, 2))
        v = self.v(value).view(b, -1, self.n_heads, self.d_k).transpose(1, 2)
        # q,k: (B,H,T,R)  v: (B,H,T,D)
        kv = torch.einsum("bhtr,bhtd->bhrd", k, v)
        k_sum = k.sum(dim=2)  # (B,H,R)
        ctx = torch.einsum("bhtr,bhrd->bhtd", q, kv)
        denom = torch.einsum("bhtr,bhr->bht", q, k_sum).unsqueeze(-1).clamp_min(1e-6)
        ctx = ctx / denom
        self.last_attn = None
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out(self.drop(ctx))
