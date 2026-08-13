"""Cross-sectional ranking and normalization (T x N panels, axis=1)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_panel(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {arr.shape}")
    return arr


def cross_sectional_rank(x: Any, *, pct: bool = False, axis: int = 1) -> np.ndarray:
    """Average-rank each cross-section. Optionally scale to [0, 1]."""
    panel = _as_panel(x)
    out = np.full(panel.shape, np.nan, dtype=np.float64)
    n = panel.shape[axis]
    if n == 0:
        return out

    # Iterate along the other axis
    other = 1 - axis if panel.ndim == 2 else 0
    for i in range(panel.shape[other]):
        row = panel[i, :] if axis == 1 else panel[:, i]
        mask = np.isfinite(row)
        k = int(mask.sum())
        if k == 0:
            continue
        vals = row[mask]
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(k, dtype=np.float64)
        # average ranks for ties
        ranks[order] = np.arange(1, k + 1, dtype=np.float64)
        # resolve ties
        sorted_vals = vals[order]
        j = 0
        while j < k:
            j2 = j + 1
            while j2 < k and sorted_vals[j2] == sorted_vals[j]:
                j2 += 1
            if j2 > j + 1:
                avg = 0.5 * (j + 1 + j2)
                ranks[order[j:j2]] = avg
            j = j2
        ranked = np.full(row.shape, np.nan, dtype=np.float64)
        ranked[mask] = ranks / k if pct else ranks
        if axis == 1:
            out[i, :] = ranked
        else:
            out[:, i] = ranked
    return out


def cross_sectional_percentile(x: Any, *, axis: int = 1) -> np.ndarray:
    """Percentile ranks in [0, 1] across the cross-section."""
    return cross_sectional_rank(x, pct=True, axis=axis)


def cross_sectional_zscore(
    x: Any,
    *,
    axis: int = 1,
    ddof: int = 1,
    clip: float | None = None,
) -> np.ndarray:
    """Cross-sectional z-score (demean / std) along ``axis`` (default 1 for T x N)."""
    panel = _as_panel(x)
    mean = np.nanmean(panel, axis=axis, keepdims=True)
    std = np.nanstd(panel, axis=axis, keepdims=True, ddof=ddof)
    std = np.where(std < 1e-12, np.nan, std)
    z = (panel - mean) / std
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


def cross_sectional_minmax(x: Any, *, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalize each cross-section to [0, 1]."""
    panel = _as_panel(x)
    lo = np.nanmin(panel, axis=axis, keepdims=True)
    hi = np.nanmax(panel, axis=axis, keepdims=True)
    span = hi - lo
    span = np.where(span < eps, np.nan, span)
    return (panel - lo) / span


def winsorize_cross_section(
    x: Any,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    axis: int = 1,
) -> np.ndarray:
    """Winsorize each cross-section at empirical quantiles."""
    panel = _as_panel(x)
    out = panel.copy()
    other = 1 - axis
    for i in range(panel.shape[other]):
        row = panel[i, :] if axis == 1 else panel[:, i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        lo, hi = np.nanquantile(row[mask], [lower, upper])
        clipped = np.clip(row, lo, hi)
        if axis == 1:
            out[i, :] = clipped
        else:
            out[:, i] = clipped
    return out
