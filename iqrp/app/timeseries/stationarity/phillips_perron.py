"""Phillips-Perron unit-root test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def phillips_perron(
    x: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
) -> AnalysisResult:
    y = as_float_array(x)
    n = y.size
    if n < 10:
        return AnalysisResult(
            method="phillips_perron",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="unit root (non-stationary)",
            alternative_hypothesis="stationary",
        )
    dy = np.diff(y)
    ylag = y[:-1]
    X = np.column_stack([np.ones(dy.size), ylag])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    resid = dy - X @ beta
    s2 = float(np.sum(resid**2) / max(len(dy) - 2, 1))
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(s2 * xtx_inv[1, 1], 0.0)))
    lags = int(np.floor(4 * (n / 100) ** 0.25))
    gamma0 = float(np.dot(resid, resid) / len(resid))
    lrv = gamma0
    for h in range(1, max(lags, 1) + 1):
        w = 1.0 - h / (lags + 1)
        gamma = float(np.dot(resid[h:], resid[:-h]) / len(resid))
        lrv += 2 * w * gamma
    lrv = max(lrv, 1e-12)
    t_stat = float(beta[1] / se) if se > 1e-15 else 0.0
    z_tau = t_stat * np.sqrt(gamma0 / lrv)
    pvalue = float(np.clip(stats.norm.cdf((z_tau + 1.6) / 0.9), 1e-6, 1 - 1e-6))
    crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    return AnalysisResult(
        method="phillips_perron",
        value="stationary" if pvalue < alpha else "non_stationary",
        statistic=float(z_tau),
        pvalue=pvalue,
        critical_values=crit,
        confidence=1.0 - alpha,
        null_hypothesis="unit root (non-stationary)",
        alternative_hypothesis="stationary",
        significant=pvalue < alpha,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"alpha": alpha, "lags": lags},
        metadata={"n": n},
    )
