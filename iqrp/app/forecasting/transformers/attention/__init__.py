"""Attention factory and package exports."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.transformers.attention.cross_asset_attention import CrossAssetAttention
from iqrp.app.forecasting.transformers.attention.flash_attention import FlashAttention
from iqrp.app.forecasting.transformers.attention.linear_attention import LinearAttention
from iqrp.app.forecasting.transformers.attention.multihead import MultiHeadAttention
from iqrp.app.forecasting.transformers.attention.performer import PerformerAttention
from iqrp.app.forecasting.transformers.attention.sparse_attention import SparseAttention
from iqrp.app.forecasting.transformers.attention.temporal_attention import (
    HierarchicalAttention,
    TemporalAttention,
)


def build_attention(
    name: str,
    d_model: int,
    n_heads: int = 4,
    dropout: float = 0.1,
    **kwargs: Any,
) -> Any:
    key = (name or "full").lower()
    if key in {"flash", "flash_attention"}:
        return FlashAttention(d_model, n_heads, dropout, chunk_size=int(kwargs.get("chunk_size", 256)))
    if key == "sparse":
        return SparseAttention(d_model, n_heads, dropout)
    if key == "linear":
        return LinearAttention(d_model, n_heads, dropout)
    if key == "performer":
        return PerformerAttention(d_model, n_heads, dropout)
    if key == "temporal":
        return TemporalAttention(d_model, n_heads, dropout)
    if key in {"hierarchical", "hierarchy"}:
        return HierarchicalAttention(d_model, n_heads, dropout)
    if key in {"cross_asset", "cross-asset"}:
        return CrossAssetAttention(d_model, n_heads, dropout)
    return MultiHeadAttention(d_model, n_heads, dropout)


__all__ = [
    "MultiHeadAttention",
    "FlashAttention",
    "SparseAttention",
    "PerformerAttention",
    "LinearAttention",
    "CrossAssetAttention",
    "TemporalAttention",
    "HierarchicalAttention",
    "build_attention",
]
