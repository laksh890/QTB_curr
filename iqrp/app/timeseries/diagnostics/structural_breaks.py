"""Structural diagnostics: breaks, shifts, heteroskedasticity, seasonality."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.decomposition.seasonal import seasonal_strength
from iqrp.app.timeseries.decomposition.stl import stl_decompose


def structural_breaks(
    x: np.ndarray | list[float],
    *,
    method: str = "pelt",
    min_size: int = 10,
) -> AnalysisResult:
    arr = as_float_array(x)
    cp = pelt_detect(arr, min_size=min_size)
    return AnalysisResult(
        method="diagnostics.structural_breaks",
        value=cp.indices,
        parameters={"method": method, "min_size": min_size},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"n_breaks": len(cp.indices), "kind": getattr(cp, "kind", "mean")},
    )


def distribution_shift(
    x: np.ndarray | list[float],
    *,
    split: int | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    arr = as_float_array(x)
    n = arr.size
    s = split if split is not None else n // 2
    a, b = arr[:s], arr[s:]
    if a.size < 5 or b.size < 5:
        return AnalysisResult(
            method="diagnostics.distribution_shift",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
        )
    stat, pvalue = stats.ks_2samp(a, b)
    return AnalysisResult(
        method="diagnostics.distribution_shift",
        value="shift" if pvalue < alpha else "stable",
        statistic=float(stat),
        pvalue=float(pvalue),
        confidence=1.0 - alpha,
        null_hypothesis="same continuous distribution",
        alternative_hypothesis="distributions differ",
        significant=pvalue < alpha,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"split": s, "alpha": alpha},
    )


def heteroskedasticity(
    x: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Breusch-Pagan-style test of residual variance vs time index."""
    y = as_float_array(x)
    n = y.size
    if n < 20:
        return AnalysisResult(
            method="diagnostics.heteroskedasticity",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
        )
    t = np.arange(n, dtype=np.float64)
    # demean
    resid = y - np.mean(y)
    e2 = resid**2
    X = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(X, e2, rcond=None)
    fitted = X @ beta
    ss_tot = float(np.sum((e2 - e2.mean()) ** 2))
    ss_res = float(np.sum((e2 - fitted) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
    lm = n * r2
    pvalue = float(1.0 - stats.chi2.cdf(lm, df=1))
    return AnalysisResult(
        method="diagnostics.heteroskedasticity",
        value="heteroskedastic" if pvalue < alpha else "homoskedastic",
        statistic=float(lm),
        pvalue=pvalue,
        null_hypothesis="homoskedastic residuals",
        alternative_hypothesis="variance depends on time",
        significant=pvalue < alpha,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"alpha": alpha},
        metadata={"r2": r2},
    )


def seasonality_diagnostics(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
) -> AnalysisResult:
    res = stl_decompose(x, period=period)
    strength = seasonal_strength(res.seasonal, res.residual)
    return AnalysisResult(
        method="diagnostics.seasonality",
        value=float(strength),
        parameters={"period": period},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"strong_seasonality": strength > 0.6},
    )


def full_diagnostics(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
    alpha: float = 0.05,
) -> dict[str, AnalysisResult]:
    return {
        "structural_breaks": structural_breaks(x),
        "distribution_shift": distribution_shift(x, alpha=alpha),
        "heteroskedasticity": heteroskedasticity(x, alpha=alpha),
        "seasonality": seasonality_diagnostics(x, period=period),
    }
