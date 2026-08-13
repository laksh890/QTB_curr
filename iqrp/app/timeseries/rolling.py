"""Rolling / expanding / adaptive window utilities (leakage-safe by default)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from iqrp.app.timeseries.base import TemporalMode, as_float_array


def rolling_apply(
    x: np.ndarray | list[float],
    window: int,
    fn: Callable[[np.ndarray], float],
    *,
    min_periods: int | None = None,
) -> np.ndarray:
    """Causal rolling window — uses only observations up to index t."""
    arr = as_float_array(x)
    n = arr.size
    w = max(int(window), 1)
    mp = max(int(min_periods if min_periods is not None else w), 1)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - w + 1)
        chunk = arr[start : i + 1]
        if chunk.size >= mp and np.isfinite(chunk).sum() >= mp:
            out[i] = float(fn(chunk[np.isfinite(chunk)]))
    return out


def expanding_apply(
    x: np.ndarray | list[float],
    fn: Callable[[np.ndarray], float],
    *,
    min_periods: int = 2,
) -> np.ndarray:
    arr = as_float_array(x)
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        chunk = arr[: i + 1]
        if chunk.size >= min_periods and np.isfinite(chunk).sum() >= min_periods:
            out[i] = float(fn(chunk[np.isfinite(chunk)]))
    return out


def multi_scale_windows(base: int, scales: tuple[int, ...] = (1, 2, 4)) -> list[int]:
    return [max(int(base * s), 2) for s in scales]


def temporal_contract(mode: str) -> TemporalMode:
    mapping = {
        "rolling": TemporalMode.ROLLING,
        "expanding": TemporalMode.EXPANDING,
        "point_in_time": TemporalMode.POINT_IN_TIME,
        "training_only": TemporalMode.TRAINING_ONLY,
        "causal": TemporalMode.CAUSAL,
        "full_sample": TemporalMode.FULL_SAMPLE,
    }
    return mapping.get(mode, TemporalMode.FULL_SAMPLE)


def incremental_mean_var(prev_n: int, prev_mean: float, prev_m2: float, x: float) -> tuple[int, float, float]:
    """Welford online update for streaming analysis."""
    n = prev_n + 1
    delta = x - prev_mean
    mean = prev_mean + delta / n
    m2 = prev_m2 + delta * (x - mean)
    return n, mean, m2
