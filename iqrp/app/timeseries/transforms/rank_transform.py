"""Rank transform wrappers."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode
from iqrp.app.timeseries.transforms import rank_transform as _rank_transform


def rank_transform(
    x: np.ndarray | list[float],
    *,
    window: int = 64,
) -> AnalysisResult:
    """Causal rolling percentile rank of the current observation in the window."""
    w = max(int(window), 2)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return AnalysisResult(
            method="transform.rank",
            value="insufficient_data",
            temporal_mode=TemporalMode.ROLLING,
            parameters={"window": w},
        )
    out = _rank_transform(x, window=w)
    return AnalysisResult(
        method="transform.rank",
        value=out,
        window=w,
        temporal_mode=TemporalMode.ROLLING,
        parameters={"window": w},
        metadata={"leakage_safe": True},
    )
