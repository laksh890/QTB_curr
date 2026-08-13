"""Group / sector demeaning neutralization for T x N signal panels."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _as_panel(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {arr.shape}")
    return arr


def demean_by_group(
    x: Any,
    groups: Sequence[Any] | np.ndarray,
    *,
    axis: int = 1,
) -> np.ndarray:
    """Subtract within-group means along the cross-section axis.

    Parameters
    ----------
    x :
        Panel shaped ``(T, N)`` when ``axis=1``.
    groups :
        Length-``N`` group labels (sector, industry, etc.).
    """
    panel = _as_panel(x)
    if axis != 1:
        raise ValueError("demean_by_group currently supports axis=1 (T x N)")
    labels = np.asarray(groups)
    if labels.shape[0] != panel.shape[1]:
        raise ValueError(
            f"groups length {labels.shape[0]} != n_assets {panel.shape[1]}"
        )
    out = panel.copy()
    uniq = {g for g in labels.tolist() if g is not None and (not isinstance(g, float) or np.isfinite(g))}
    # Also keep NaN/None groups as their own bucket skipped for demeaning
    for g in uniq:
        mask = labels == g
        if not np.any(mask):
            continue
        sub = out[:, mask]
        mu = np.nanmean(sub, axis=1, keepdims=True)
        out[:, mask] = sub - mu
    return out


def neutralize_market(x: Any, *, axis: int = 1) -> np.ndarray:
    """Cross-sectional demean (market neutralization)."""
    panel = _as_panel(x)
    mu = np.nanmean(panel, axis=axis, keepdims=True)
    return panel - mu


def neutralize_weighted(
    x: Any,
    weights: Any,
    *,
    axis: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Subtract weighted cross-sectional mean (e.g. market-cap weighted)."""
    panel = _as_panel(x)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = np.broadcast_to(w.reshape(1, -1), panel.shape)
    elif w.shape != panel.shape:
        raise ValueError(f"weights shape {w.shape} incompatible with {panel.shape}")
    w = np.where(np.isfinite(w) & np.isfinite(panel), np.maximum(w, 0.0), 0.0)
    num = np.nansum(panel * w, axis=axis, keepdims=True)
    den = np.nansum(w, axis=axis, keepdims=True)
    mu = num / np.maximum(den, eps)
    return panel - mu


def neutralize_multi_group(
    x: Any,
    group_sets: Sequence[Sequence[Any] | np.ndarray],
    *,
    n_passes: int = 2,
) -> np.ndarray:
    """Iteratively demean by multiple group taxonomies (sector then industry, …)."""
    out = _as_panel(x)
    for _ in range(max(1, int(n_passes))):
        for groups in group_sets:
            out = demean_by_group(out, groups, axis=1)
    return out
