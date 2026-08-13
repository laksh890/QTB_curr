"""Normalization transform wrappers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode
from iqrp.app.timeseries.transforms import normalize as _normalize


def normalize(
    x: np.ndarray | list[float],
    *,
    method: Literal["zscore", "robust", "minmax"] = "zscore",
    window: int = 64,
) -> AnalysisResult:
    """Causal rolling normalization (z-score, robust, or min-max)."""
    w = max(int(window), 2)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return AnalysisResult(
            method=f"transform.normalize.{method}",
            value="insufficient_data",
            temporal_mode=TemporalMode.ROLLING,
            parameters={"method": method, "window": w},
        )
    out = _normalize(x, method=method, window=w)
    return AnalysisResult(
        method=f"transform.normalize.{method}",
        value=out,
        window=w,
        temporal_mode=TemporalMode.ROLLING,
        parameters={"method": method, "window": w},
        metadata={"leakage_safe": True},
    )


def zscore_normalize(x: np.ndarray | list[float], *, window: int = 64) -> AnalysisResult:
    """Causal rolling z-score normalization."""
    return normalize(x, method="zscore", window=window)


def robust_normalize(x: np.ndarray | list[float], *, window: int = 64) -> AnalysisResult:
    """Causal rolling robust (median/IQR) normalization."""
    return normalize(x, method="robust", window=window)


def minmax_normalize(x: np.ndarray | list[float], *, window: int = 64) -> AnalysisResult:
    """Causal rolling min-max normalization to [0, 1]."""
    return normalize(x, method="minmax", window=window)
