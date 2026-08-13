"""Portfolio turnover measurement."""

from __future__ import annotations

from typing import Any

import numpy as np


def turnover_series(weights: Any, *, half: bool = True) -> np.ndarray:
    """Per-period turnover from a weight path.

    ``half=True`` uses the standard 0.5 * L1 definition (one-way).
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    t = w.shape[0]
    out = np.zeros(t, dtype=np.float64)
    scale = 0.5 if half else 1.0
    out[0] = scale * float(np.nansum(np.abs(w[0])))
    if t > 1:
        out[1:] = scale * np.nansum(np.abs(np.diff(w, axis=0)), axis=1)
    return out


def average_turnover(weights: Any, *, half: bool = True) -> float:
    """Mean per-period turnover."""
    series = turnover_series(weights, half=half)
    if series.size == 0:
        return 0.0
    return float(np.nanmean(series))


def annualized_turnover(
    weights: Any,
    *,
    periods_per_year: float = 252.0,
    half: bool = True,
) -> float:
    """Approximate annualized turnover = mean period turnover × periods/year."""
    return average_turnover(weights, half=half) * float(periods_per_year)
