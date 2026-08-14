"""Sector / industry adjustment utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.alpha.cross_section.neutralization import demean_by_group, neutralize_weighted
from iqrp.app.alpha.cross_section.ranking import cross_sectional_zscore


def sector_neutral_zscore(
    x: Any,
    sectors: Sequence[Any] | np.ndarray,
    *,
    clip: float | None = 3.0,
) -> np.ndarray:
    """Demean by sector then cross-sectional z-score."""
    neut = demean_by_group(x, sectors, axis=1)
    return cross_sectional_zscore(neut, axis=1, clip=clip)


def industry_neutralize(
    x: Any,
    industries: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Alias for industry demeaning."""
    return demean_by_group(x, industries, axis=1)


def sector_relative_ranks(
    x: Any,
    sectors: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Percentile rank within each sector (NaN outside finite observations)."""
    panel = np.asarray(x, dtype=np.float64)
    if panel.ndim == 1:
        panel = panel.reshape(1, -1)
    labels = np.asarray(sectors)
    if labels.shape[0] != panel.shape[1]:
        raise ValueError("sectors length must equal n_assets")
    out = np.full(panel.shape, np.nan, dtype=np.float64)
    for g in {s for s in labels.tolist()}:
        mask = labels == g
        cols = np.where(mask)[0]
        if cols.size == 0:
            continue
        sub = panel[:, cols]
        for i in range(sub.shape[0]):
            row = sub[i]
            m = np.isfinite(row)
            k = int(m.sum())
            if k == 0:
                continue
            order = np.argsort(row[m], kind="mergesort")
            ranks = np.empty(k, dtype=np.float64)
            ranks[order] = (np.arange(1, k + 1, dtype=np.float64) - 0.5) / k
            ranked = np.full(row.shape, np.nan)
            ranked[m] = ranks
            out[i, cols] = ranked
    return out


def cap_weighted_sector_neutral(
    x: Any,
    sectors: Sequence[Any] | np.ndarray,
    market_caps: Any,
) -> np.ndarray:
    """Within-sector demean using market-cap weights, then global demean."""
    panel = np.asarray(x, dtype=np.float64)
    if panel.ndim == 1:
        panel = panel.reshape(1, -1)
    caps = np.asarray(market_caps, dtype=np.float64)
    if caps.ndim == 1:
        caps = np.broadcast_to(caps.reshape(1, -1), panel.shape)
    labels = np.asarray(sectors)
    out = panel.copy()
    for g in {s for s in labels.tolist()}:
        mask = labels == g
        if not np.any(mask):
            continue
        sub = out[:, mask]
        w = caps[:, mask]
        out[:, mask] = neutralize_weighted(sub, w, axis=1)
    return out
