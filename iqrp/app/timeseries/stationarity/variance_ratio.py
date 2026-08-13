"""Lo-MacKinlay variance ratio test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def variance_ratio(
    x: np.ndarray | list[float],
    *,
    lags: int = 2,
    alpha: float = 0.05,
) -> AnalysisResult:
    """H0: series is a random walk (VR=1)."""
    y = as_float_array(x)
    r = np.diff(y)
    n = r.size
    q = max(int(lags), 2)
    if n < q * 4:
        return AnalysisResult(
            method="variance_ratio",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="random walk (VR=1)",
            alternative_hypothesis="VR != 1 (mean reversion or momentum)",
        )
    mu = float(np.mean(r))
    var1 = float(np.sum((r - mu) ** 2) / (n - 1))
    rq = np.array([np.sum(r[i : i + q]) for i in range(n - q + 1)], dtype=np.float64)
    varq = float(np.sum((rq - q * mu) ** 2) / (q * (n - q + 1)))
    vr = varq / var1 if var1 > 1e-15 else 1.0
    theta = (2 * (2 * q - 1) * (q - 1)) / (3 * q * n)
    z = (vr - 1.0) / np.sqrt(max(theta, 1e-15))
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    return AnalysisResult(
        method="variance_ratio",
        value=float(vr),
        statistic=float(z),
        pvalue=pvalue,
        confidence=1.0 - alpha,
        confidence_interval=(float(vr - 1.96 * np.sqrt(theta)), float(vr + 1.96 * np.sqrt(theta))),
        null_hypothesis="random walk (VR=1)",
        alternative_hypothesis="VR != 1 (mean reversion or momentum)",
        significant=pvalue < alpha,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"lags": q, "alpha": alpha},
        metadata={"n": n, "interpretation": "VR<1 mean-reverting; VR>1 trending"},
    )
