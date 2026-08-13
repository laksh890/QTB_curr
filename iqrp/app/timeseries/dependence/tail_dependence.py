"""Empirical upper/lower tail dependence coefficients."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def empirical_tail_dependence(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    quantile: float = 0.05,
) -> AnalysisResult:
    """Empirical lower & upper tail dependence at given quantile (FULL_SAMPLE).

    Lower: P(U<q, V<q) / q using empirical ranks (pseudo-observations).
    Upper: P(U>1-q, V>1-q) / q.
    """
    a = as_float_array(x)
    b = as_float_array(y)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    q = float(np.clip(quantile, 1e-3, 0.5))
    if n < max(20, int(1.0 / q) * 2):
        return AnalysisResult(
            method="empirical_tail_dependence",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="tail dependence = 0 (asymptotic independence)",
            alternative_hypothesis="positive tail dependence",
            parameters={"quantile": q},
        )
    # empirical CDF via ranks / (n+1)
    u = (np.argsort(np.argsort(a)) + 1.0) / (n + 1.0)
    v = (np.argsort(np.argsort(b)) + 1.0) / (n + 1.0)
    lower = float(np.mean((u <= q) & (v <= q)) / q)
    upper = float(np.mean((u >= 1.0 - q) & (v >= 1.0 - q)) / q)
    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, 0.0, 1.0))
    return AnalysisResult(
        method="empirical_tail_dependence",
        value={"lower": lower, "upper": upper},
        statistic=float(max(lower, upper)),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="tail dependence = 0 (asymptotic independence)",
        alternative_hypothesis="positive tail dependence",
        significant=max(lower, upper) > q * 2,  # heuristic vs independence baseline ~q
        parameters={"quantile": q},
        metadata={"n": n},
    )
