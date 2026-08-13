"""KPSS stationarity test."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def kpss(
    x: np.ndarray | list[float],
    *,
    regression: str = "c",
    alpha: float = 0.05,
) -> AnalysisResult:
    y = as_float_array(x)
    n = y.size
    if n < 10:
        return AnalysisResult(
            method="kpss",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="stationary",
            alternative_hypothesis="unit root (non-stationary)",
        )
    if regression == "ct":
        t = np.arange(n, dtype=np.float64)
        X = np.column_stack([np.ones(n), t])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        crit = {"10%": 0.119, "5%": 0.146, "2.5%": 0.176, "1%": 0.216}
    else:
        resid = y - np.mean(y)
        crit = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    S = np.cumsum(resid)
    # Newey-West long-run variance
    lags = int(np.floor(4 * (n / 100) ** 0.25))
    gamma0 = float(np.dot(resid, resid) / n)
    lrv = gamma0
    for h in range(1, lags + 1):
        w = 1.0 - h / (lags + 1)
        gamma = float(np.dot(resid[h:], resid[:-h]) / n)
        lrv += 2 * w * gamma
    lrv = max(lrv, 1e-12)
    eta = float(np.sum(S**2) / (n**2 * lrv))
    # interpolate p-value from critical values
    pvalue = _kpss_pvalue(eta, crit)
    return AnalysisResult(
        method="kpss",
        value="stationary" if pvalue > alpha else "non_stationary",
        statistic=eta,
        pvalue=pvalue,
        critical_values=crit,
        confidence=1.0 - alpha,
        null_hypothesis="stationary",
        alternative_hypothesis="unit root (non-stationary)",
        significant=pvalue < alpha,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"regression": regression, "alpha": alpha},
        metadata={"n": n, "lags": lags},
    )


def _kpss_pvalue(stat: float, crit: dict[str, float]) -> float:
    # piecewise linear in log-p space
    pts = sorted(((v, float(k.strip("%")) / 100.0) for k, v in crit.items()), key=lambda z: z[0])
    if stat <= pts[0][0]:
        return min(0.99, pts[0][1] * 2)
    if stat >= pts[-1][0]:
        return max(0.001, pts[-1][1] / 2)
    for i in range(len(pts) - 1):
        x0, p0 = pts[i]
        x1, p1 = pts[i + 1]
        if x0 <= stat <= x1:
            t = (stat - x0) / (x1 - x0 + 1e-15)
            return float(p0 + t * (p1 - p0))
    return 0.05
