"""Differencing transform wrappers."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode
from iqrp.app.timeseries.transforms import TimeSeriesTransformer
from iqrp.app.timeseries.transforms import differencing as _differencing


def differencing(
    x: np.ndarray | list[float],
    *,
    order: int = 1,
) -> AnalysisResult:
    """Causal lag-``order`` differencing Δ_k x_t = x_t - x_{t-k}."""
    k = max(int(order), 1)
    out = _differencing(x, order=k)
    return AnalysisResult(
        method="transform.differencing",
        value=out,
        temporal_mode=TemporalMode.CAUSAL,
        parameters={"order": k},
        metadata={"leakage_safe": True},
    )


def seasonal_differencing(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
) -> AnalysisResult:
    """Causal seasonal differencing with lag ``period``."""
    p = max(int(period), 1)
    tf = TimeSeriesTransformer(method="seasonal_diff", period=p)
    out = tf.fit_transform(x)
    return AnalysisResult(
        method="transform.seasonal_differencing",
        value=out,
        temporal_mode=TemporalMode.CAUSAL,
        parameters={"period": p},
        metadata={"leakage_safe": True},
    )
