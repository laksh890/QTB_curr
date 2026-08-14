"""Granger causality F-test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def granger_causality(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    max_lag: int = 2,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Test whether x Granger-causes y (FULL_SAMPLE).

    Restricted model: y_t ~ lags(y); unrestricted: y_t ~ lags(y) + lags(x).
    """
    a = as_float_array(x)
    b = as_float_array(y)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    p = max(int(max_lag), 1)
    if n < 3 * p + 5:
        return AnalysisResult(
            method="granger_causality",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="x does not Granger-cause y",
            alternative_hypothesis="x Granger-causes y",
            parameters={"max_lag": p, "alpha": alpha},
        )
    # build design
    rows = n - p
    y_t = b[p:]
    Ylags = np.column_stack([b[p - i : n - i] for i in range(1, p + 1)])
    Xlags = np.column_stack([a[p - i : n - i] for i in range(1, p + 1)])
    X_r = np.column_stack([np.ones(rows), Ylags])
    X_u = np.column_stack([np.ones(rows), Ylags, Xlags])

    def _ssr(X: np.ndarray, yy: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        return float(np.sum(resid**2))

    ssr_r = _ssr(X_r, y_t)
    ssr_u = _ssr(X_u, y_t)
    df1 = p
    df2 = rows - X_u.shape[1]
    if df2 <= 0 or ssr_u < 1e-18:
        return AnalysisResult(
            method="granger_causality",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="x does not Granger-cause y",
            alternative_hypothesis="x Granger-causes y",
            parameters={"max_lag": p, "alpha": alpha},
        )
    F = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    pvalue = float(stats.f.sf(F, df1, df2))
    sig = pvalue < alpha
    return AnalysisResult(
        method="granger_causality",
        value="x_granger_causes_y" if sig else "no_granger_causality",
        statistic=float(F),
        pvalue=pvalue,
        confidence=1.0 - alpha,
        null_hypothesis="x does not Granger-cause y",
        alternative_hypothesis="x Granger-causes y",
        significant=sig,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"max_lag": p, "alpha": alpha},
        metadata={
            "n": n,
            "ssr_restricted": ssr_r,
            "ssr_unrestricted": ssr_u,
            "df1": df1,
            "df2": df2,
        },
    )
