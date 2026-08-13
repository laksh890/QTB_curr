"""Distance correlation (Szekely et al.)."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def distance_correlation(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> AnalysisResult:
    """Sample distance correlation dCor(X, Y) in [0, 1] (FULL_SAMPLE)."""
    a = as_float_array(x)
    b = as_float_array(y)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 5:
        return AnalysisResult(
            method="distance_correlation",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="dCor=0 (independence)",
            alternative_hypothesis="dCor>0 (dependence, possibly nonlinear)",
        )
    A = _centered_distance(a)
    B = _centered_distance(b)
    dcov2 = float(np.mean(A * B))
    dvarx = float(np.mean(A * A))
    dvary = float(np.mean(B * B))
    if dvarx < 1e-18 or dvary < 1e-18:
        dcor = 0.0
    else:
        # R^2 = V^2_xy / sqrt(V^2_x V^2_y)
        dcor = float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvarx * dvary)))
    dcor = float(np.clip(dcor, 0.0, 1.0))
    return AnalysisResult(
        method="distance_correlation",
        value=dcor,
        statistic=dcor,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="dCor=0 (independence)",
        alternative_hypothesis="dCor>0 (dependence, possibly nonlinear)",
        significant=dcor > 0.1,
        parameters={},
        metadata={"n": n, "distance_covariance": float(np.sqrt(max(dcov2, 0.0)))},
    )


def _centered_distance(v: np.ndarray) -> np.ndarray:
    D = np.abs(v[:, None] - v[None, :])
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand = D.mean()
    return D - row_mean - col_mean + grand
