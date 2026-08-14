"""Augmented Dickey-Fuller stationarity test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def adf(
    x: np.ndarray | list[float],
    *,
    max_lag: int | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    y = as_float_array(x)
    n = y.size
    if n < 10:
        return AnalysisResult(
            method="adf",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="unit root (non-stationary)",
            alternative_hypothesis="stationary",
        )
    dy = np.diff(y)
    y_lag = y[:-1]
    # select lag via Schwert rule
    k = max_lag if max_lag is not None else int(np.floor(12 * (n / 100) ** 0.25))
    k = int(np.clip(k, 0, max(n // 5, 0)))
    X_cols = [np.ones(dy.size - k), y_lag[k:]]
    for i in range(1, k + 1):
        X_cols.append(dy[k - i : dy.size - i])
    X = np.column_stack(X_cols)
    yy = dy[k:]
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    s2 = float(np.sum(resid**2) / max(len(yy) - X.shape[1], 1))
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(s2 * xtx_inv[1, 1], 0.0)))
    stat = float(beta[1] / se) if se > 1e-15 else 0.0
    # MacKinnon approximate p-value (constant model)
    pvalue = _mackinnon_pvalue(stat)
    crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    return AnalysisResult(
        method="adf",
        value="stationary" if pvalue < alpha else "non_stationary",
        statistic=stat,
        pvalue=pvalue,
        critical_values=crit,
        confidence=1.0 - alpha,
        null_hypothesis="unit root (non-stationary)",
        alternative_hypothesis="stationary",
        significant=pvalue < alpha,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"max_lag": k, "alpha": alpha},
        metadata={"n": n},
    )


def _mackinnon_pvalue(stat: float) -> float:
    # rough mapping via normal CDF of shifted statistic
    # calibrated to approximate MacKinnon for constant-only ADF
    z = (stat + 1.6) / 0.9
    return float(np.clip(stats.norm.cdf(z), 1e-6, 1 - 1e-6))
