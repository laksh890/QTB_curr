"""Transformer encoder stack."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.attention import build_attention

try:
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    nn = object  # type: ignore[assignment]


class TransformerEncoderLayer(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        attention_type: str = "full",
    ) -> None:
        if has_torch():
            super().__init__()
        self.self_attn = build_attention(attention_type, d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x: Any, mask: Any = None) -> Any:
        # TemporalAttention / CrossAsset return already-normalized residual; MultiHead needs wrap
        if hasattr(self.self_attn, "attn") and not hasattr(self.self_attn, "q"):
            h = self.self_attn(x, mask=mask)
        else:
            h = self.n1(x + self.self_attn(x, x, x, mask=mask))
        return self.n2(h + self.ff(h))


class TransformerEncoder(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        attention_type: str = "full",
    ) -> None:
        if has_torch():
            super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout, attention_type)
                for _ in range(max(num_layers, 1))
            ]
        )

    def forward(self, x: Any, mask: Any = None) -> Any:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
