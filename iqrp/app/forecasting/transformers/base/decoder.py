"""Transformer decoder stack with cross-attention."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.attention.multihead import MultiHeadAttention

try:
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    nn = object  # type: ignore[assignment]


class TransformerDecoderLayer(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, n_heads: int = 4, ffn_dim: int = 128, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x: Any, memory: Any, tgt_mask: Any = None, mem_mask: Any = None) -> Any:
        x = self.n1(x + self.self_attn(x, x, x, mask=tgt_mask))
        x = self.n2(x + self.cross_attn(x, memory, memory, mask=mem_mask))
        return self.n3(x + self.ff(x))


class TransformerDecoder(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        if has_torch():
            super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, ffn_dim, dropout) for _ in range(max(num_layers, 1))]
        )

    def forward(self, x: Any, memory: Any, tgt_mask: Any = None, mem_mask: Any = None) -> Any:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return x
