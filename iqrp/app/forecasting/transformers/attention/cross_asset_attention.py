"""Cross-asset attention for multi-asset / cross-sectional forecasting."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.transformers.attention.multihead import MultiHeadAttention

try:
    from torch import nn
except Exception:  # pragma: no cover
    nn = object  # type: ignore[assignment]


class CrossAssetAttention(nn.Module if has_torch() else object):  # type: ignore[misc]
    """Attend across the asset dimension: input (B, A, T, D) -> same."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Any, mask: Any = None) -> Any:
        # x: (B, A, T, D) — pool time then attend assets, or attend per timestep
        b, a, t, d = x.shape
        # per-timestep cross-asset
        xt = x.permute(0, 2, 1, 3).reshape(b * t, a, d)
        out = self.attn(xt, xt, xt, mask=mask)
        out = out.view(b, t, a, d).permute(0, 2, 1, 3)
        return self.norm(x + out)
