"""Multiple Seasonal-Trend decomposition (MSTL)."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import DecompositionResult, TemporalMode, as_float_array
from iqrp.app.timeseries.decomposition.stl import stl_decompose


def mstl_decompose(
    x: np.ndarray | list[float],
    *,
    periods: tuple[int, ...] = (24, 168),
    robust: bool = False,
) -> DecompositionResult:
    arr = as_float_array(x)
    remaining = arr.copy()
    seasonals = []
    trend = np.zeros_like(arr)
    for p in periods:
        p = max(int(p), 2)
        if arr.size < p * 2:
            continue
        res = stl_decompose(remaining, period=p, robust=robust)
        seasonals.append(res.seasonal)
        remaining = remaining - res.seasonal
        trend = res.trend
    seasonal = np.sum(seasonals, axis=0) if seasonals else np.zeros_like(arr)
    residual = arr - trend - seasonal
    return DecompositionResult(
        method="mstl",
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        observed=arr,
        model="additive",
        parameters={"periods": list(periods), "robust": robust},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"n_seasons": len(seasonals), "components": [s.tolist() for s in seasonals]},
    )
