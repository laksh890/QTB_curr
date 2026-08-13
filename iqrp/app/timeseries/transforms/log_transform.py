"""Log transform wrappers."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.transforms import TimeSeriesTransformer


def log_transform(
    x: np.ndarray | list[float],
    *,
    clip_min: float = 1e-12,
) -> AnalysisResult:
    """Elementwise log with positivity clip (POINT_IN_TIME / causal)."""
    arr = as_float_array(x)
    if arr.size == 0:
        return AnalysisResult(
            method="transform.log",
            value="insufficient_data",
            temporal_mode=TemporalMode.POINT_IN_TIME,
            parameters={"clip_min": clip_min},
        )
    out = np.log(np.clip(arr, clip_min, None))
    return AnalysisResult(
        method="transform.log",
        value=out,
        temporal_mode=TemporalMode.POINT_IN_TIME,
        parameters={"clip_min": clip_min},
        metadata={"leakage_safe": True},
    )


def log_via_transformer(x: np.ndarray | list[float]) -> np.ndarray:
    """Array-only log via ``TimeSeriesTransformer(method='log')``."""
    return TimeSeriesTransformer(method="log").fit_transform(x)
