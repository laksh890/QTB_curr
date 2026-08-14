"""Cointegration tests (Engle-Granger and simplified Johansen)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.stationarity.adf import adf


def engle_granger(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Engle–Granger two-step cointegration test (FULL_SAMPLE)."""
    a = as_float_array(x)
    b = as_float_array(y)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 20:
        return AnalysisResult(
            method="engle_granger",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no cointegration (residual has unit root)",
            alternative_hypothesis="cointegrated (residual stationary)",
            parameters={"alpha": alpha},
        )
    # step 1: OLS y = c + beta x
    X = np.column_stack([np.ones(n), a])
    beta, *_ = np.linalg.lstsq(X, b, rcond=None)
    resid = b - X @ beta
    # step 2: ADF on residuals (no constant in cointegrating residual ADF is common;
    # we reuse adf which includes constant — still informative)
    adf_res = adf(resid, alpha=alpha)
    # Engle-Granger critical values (constant case, MacKinnon approx)
    eg_stat = float(adf_res.statistic) if adf_res.statistic is not None else np.nan
    crit = {"1%": -3.90, "5%": -3.34, "10%": -3.04}
    # decide via critical value at requested alpha
    if alpha <= 0.01:
        cv = crit["1%"]
    elif alpha <= 0.05:
        cv = crit["5%"]
    else:
        cv = crit["10%"]
    coint = bool(np.isfinite(eg_stat) and eg_stat < cv)
    pvalue = _eg_pvalue(eg_stat)
    return AnalysisResult(
        method="engle_granger",
        value="cointegrated" if coint else "not_cointegrated",
        statistic=eg_stat,
        pvalue=pvalue,
        critical_values=crit,
        confidence=1.0 - alpha,
        null_hypothesis="no cointegration (residual has unit root)",
        alternative_hypothesis="cointegrated (residual stationary)",
        significant=coint,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"alpha": alpha, "beta": float(beta[1]), "intercept": float(beta[0])},
        metadata={"n": n, "residual_adf_pvalue": adf_res.pvalue},
    )


def johansen_trace(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    lag: int = 1,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Simplified 2-series Johansen trace test (FULL_SAMPLE).

    Implements the VECM(ℓ) eigenvalue approach for bivariate systems only.
    """
    a = as_float_array(x)
    b = as_float_array(y)
    n = min(a.size, b.size)
    Z = np.column_stack([a[:n], b[:n]])
    mask = np.isfinite(Z).all(axis=1)
    Z = Z[mask]
    T = Z.shape[0]
    k = max(int(lag), 1)
    if 30 + k > T:
        return AnalysisResult(
            method="johansen_trace",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="r=0 (no cointegration)",
            alternative_hypothesis="r≥1 (at least one cointegrating relation)",
            parameters={"lag": k, "alpha": alpha},
        )
    dZ = np.diff(Z, axis=0)
    Z_lag = Z[:-1]
    # build lagged differences for VAR
    rows = dZ.shape[0] - k
    dy = dZ[k:]
    zlag = Z_lag[k:]
    # regressors: constant + Z_{t-1} + dZ lags
    cols = [np.ones(rows)]
    cols.append(zlag[:, 0])
    cols.append(zlag[:, 1])
    for i in range(1, k + 1):
        cols.append(dZ[k - i : k - i + rows, 0])
        cols.append(dZ[k - i : k - i + rows, 1])
    np.column_stack(cols)
    # residuals from dy ~ X without Z_lag (restricted) and with — Johansen uses
    # R0t = residuals of dY on lags only; R1t = residuals of Y_{t-1} on lags only
    X_lags = np.column_stack([np.ones(rows)] + cols[3:])
    B0, *_ = np.linalg.lstsq(X_lags, dy, rcond=None)
    R0 = dy - X_lags @ B0
    B1, *_ = np.linalg.lstsq(X_lags, zlag, rcond=None)
    R1 = zlag - X_lags @ B1

    S00 = (R0.T @ R0) / rows
    S11 = (R1.T @ R1) / rows
    S01 = (R0.T @ R1) / rows
    S10 = S01.T
    try:
        S11_inv = np.linalg.pinv(S11)
        M = np.linalg.solve(S00, S01 @ S11_inv @ S10)
        eigvals = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
        eigvals = np.clip(eigvals, 0.0, 1.0 - 1e-12)
    except np.linalg.LinAlgError:
        return AnalysisResult(
            method="johansen_trace",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="r=0 (no cointegration)",
            alternative_hypothesis="r≥1 (at least one cointegrating relation)",
            parameters={"lag": k, "alpha": alpha},
            metadata={"reason": "linalg_error"},
        )
    # trace statistic for r=0: -T Σ log(1-λ_i)
    trace0 = float(-rows * np.sum(np.log(1.0 - eigvals)))
    # Osterwald-Lenum approx critical values for p-r=2 (constant)
    crit = {"1%": 19.94, "5%": 15.49, "10%": 13.43}
    # rough p-value via chi2 mapping
    pvalue = float(stats.chi2.sf(trace0, df=4))
    coint = trace0 > crit["5%"]
    return AnalysisResult(
        method="johansen_trace",
        value={"rank_ge_1": coint, "eigenvalues": eigvals.tolist(), "trace_r0": trace0},
        statistic=trace0,
        pvalue=float(np.clip(pvalue, 0.0, 1.0)),
        critical_values=crit,
        confidence=1.0 - alpha,
        null_hypothesis="r=0 (no cointegration)",
        alternative_hypothesis="r≥1 (at least one cointegrating relation)",
        significant=coint,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"lag": k, "alpha": alpha},
        metadata={"n": T, "rows": rows},
    )


def _eg_pvalue(stat: float) -> float:
    """Approximate EG p-value from MacKinnon-style critical points."""
    if not np.isfinite(stat):
        return 1.0
    # interpolate in probability space between tabulated critical values
    points = [(-3.90, 0.01), (-3.34, 0.05), (-3.04, 0.10), (-2.50, 0.50), (-1.50, 0.90)]
    if stat <= points[0][0]:
        return float(max(1e-6, 0.01 * np.exp(stat - points[0][0])))
    if stat >= points[-1][0]:
        return float(min(1.0 - 1e-6, 0.90 + 0.05 * (stat - points[-1][0])))
    for (s0, p0), (s1, p1) in zip(points[:-1], points[1:]):
        if s0 <= stat <= s1:
            w = (stat - s0) / (s1 - s0)
            return float(p0 + w * (p1 - p0))
    return 0.5
