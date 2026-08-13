"""Attention masking utilities for transformers."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]


def causal_mask(seq_len: int, device: Any = None) -> Any:
    if not has_torch():
        return None
    m = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return m


def padding_mask(lengths: Any, max_len: int) -> Any:
    if not has_torch():
        return None
    b = lengths.shape[0]
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(b, -1)
    return idx >= lengths.unsqueeze(1)


def local_attention_mask(seq_len: int, window: int, device: Any = None) -> Any:
    if not has_torch():
        return None
    idx = torch.arange(seq_len, device=device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return dist > max(int(window), 1)


def regime_mask(regime_ids: Any, allow_cross: bool = False) -> Any:
    """Block attention across regimes when allow_cross is False."""
    if not has_torch() or allow_cross:
        return None
    same = regime_ids.unsqueeze(1) == regime_ids.unsqueeze(2)
    return ~same


def combine_masks(*masks: Any) -> Any:
    out = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out | m)
    return out


def apply_mask_to_scores(scores: Any, mask: Any, fill: float = -1e9) -> Any:
    if mask is None:
        return scores
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return scores.masked_fill(mask, fill)
