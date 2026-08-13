"""Positional encodings: sinusoidal, learned, rotary."""

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


class SinusoidalPositionalEncoding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, max_len: int = 10240) -> None:
        if has_torch():
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / max(d_model, 1)))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Any) -> Any:
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEncoding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, max_len: int = 10240) -> None:
        if has_torch():
            super().__init__()
            self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: Any) -> Any:
        b, t, _ = x.shape
        idx = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        return x + self.emb(idx)


class RotaryPositionalEncoding(nn.Module if has_torch() else object):  # type: ignore[misc]
    """Apply RoPE to query/key tensors of shape (B, H, T, D)."""

    def __init__(self, dim: int, max_len: int = 10240) -> None:
        if has_torch():
            super().__init__()
            inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / max(dim, 1)))
            t = torch.arange(max_len).float()
            freqs = torch.outer(t, inv)
            self.register_buffer("cos", freqs.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin", freqs.sin()[None, None, :, :], persistent=False)
            self.dim = dim

    def forward(self, q: Any, k: Any) -> tuple[Any, Any]:
        t = q.size(-2)
        cos = self.cos[:, :, :t, : q.size(-1) // 2]
        sin = self.sin[:, :, :t, : q.size(-1) // 2]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: Any, cos: Any, sin: Any) -> Any:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


def build_positional(name: str, d_model: int, max_len: int = 10240) -> Any:
    key = (name or "sinusoidal").lower()
    if key == "learned":
        return LearnedPositionalEncoding(d_model, max_len)
    if key == "rotary":
        return RotaryPositionalEncoding(d_model, max_len)
    return SinusoidalPositionalEncoding(d_model, max_len)
