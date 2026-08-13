"""Autocorrelation function (ACF) with Bartlett significance bands."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def acf(
    x: np.ndarray | list[float],
    *,
    nlags: int | None = None,
    alpha: float = 0.05,
    demean: bool = True,
) -> AnalysisResult:
    """Sample ACF for lags 0..nlags (FULL_SAMPLE research estimator)."""
    y = as_float_array(x)
    n = int(np.sum(np.isfinite(y)))
    if n < 3:
        return AnalysisResult(
            method="acf",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="rho(k)=0 for all k>0 (white noise)",
            alternative_hypothesis="serial correlation at some lag",
            parameters={"nlags": nlags, "alpha": alpha},
        )
    y = y[np.isfinite(y)]
    n = y.size
    max_lag = int(nlags) if nlags is not None else min(n - 1, int(np.floor(10 * np.log10(n))))
    max_lag = int(np.clip(max_lag, 1, n - 1))
    if demean:
        y = y - np.mean(y)
    c0 = float(np.dot(y, y) / n)
    if c0 < 1e-18:
        lags = np.arange(max_lag + 1, dtype=np.float64)
        acf_vals = np.ones(max_lag + 1)
        return AnalysisResult(
            method="acf",
            value=acf_vals,
            statistic=0.0,
            pvalue=1.0,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="rho(k)=0 for all k>0 (white noise)",
            alternative_hypothesis="serial correlation at some lag",
            parameters={"nlags": max_lag, "alpha": alpha},
            metadata={"lags": lags.tolist(), "constant_series": True},
        )
    acf_vals = np.empty(max_lag + 1, dtype=np.float64)
    acf_vals[0] = 1.0
    for k in range(1, max_lag + 1):
        acf_vals[k] = float(np.dot(y[:-k], y[k:]) / n) / c0

    # Bartlett bands for white-noise null (lags > 0)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = np.empty(max_lag + 1, dtype=np.float64)
    se[0] = 0.0
    for k in range(1, max_lag + 1):
        # Bartlett: Var(r_k) ≈ (1 + 2 Σ_{j=1}^{k-1} r_j^2) / n under MA(k-1)
        if k == 1:
            se[k] = 1.0 / np.sqrt(n)
        else:
            se[k] = np.sqrt((1.0 + 2.0 * np.sum(acf_vals[1:k] ** 2)) / n)
    lower = -z * se
    upper = z * se
    lower[0], upper[0] = 1.0, 1.0

    # Ljung-Box style omnibus on lags 1..max_lag
    lb_stat = float(n * (n + 2) * np.sum((acf_vals[1:] ** 2) / (n - np.arange(1, max_lag + 1))))
    pvalue = float(stats.chi2.sf(lb_stat, max_lag))
    significant = bool(np.any(np.abs(acf_vals[1:]) > upper[1:]))

    return AnalysisResult(
        method="acf",
        value=acf_vals,
        statistic=lb_stat,
        pvalue=pvalue,
        confidence=1.0 - alpha,
        confidence_interval=(float(lower[1]), float(upper[1])) if max_lag >= 1 else None,
        null_hypothesis="rho(k)=0 for all k>0 (white noise)",
        alternative_hypothesis="serial correlation at some lag",
        significant=significant,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"nlags": max_lag, "alpha": alpha, "demean": demean},
        metadata={
            "lags": list(range(max_lag + 1)),
            "lower_band": lower.tolist(),
            "upper_band": upper.tolist(),
            "standard_errors": se.tolist(),
            "n": n,
        },
    )


def rolling_acf(
    x: np.ndarray | list[float],
    *,
    window: int = 64,
    lag: int = 1,
    min_periods: int | None = None,
) -> AnalysisResult:
    """Causal rolling single-lag autocorrelation (ROLLING / online-safe)."""
    y = as_float_array(x)
    n = y.size
    w = max(int(window), 3)
    k = max(int(lag), 1)
    mp = max(int(min_periods if min_periods is not None else w), k + 2)
    if n < mp:
        return AnalysisResult(
            method="rolling_acf",
            value="insufficient_data",
            temporal_mode=TemporalMode.ROLLING,
            null_hypothesis="rho(lag)=0 within each window",
            alternative_hypothesis="nonzero rolling autocorrelation",
            parameters={"window": w, "lag": k},
        )
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - w + 1)
        chunk = y[start : i + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size < mp:
            continue
        m = finite.size
        z = finite - np.mean(finite)
        c0 = float(np.dot(z, z) / m)
        if c0 < 1e-18 or k >= m:
            out[i] = 0.0
        else:
            out[i] = float(np.dot(z[:-k], z[k:]) / m) / c0
    return AnalysisResult(
        method="rolling_acf",
        value=out,
        window=w,
        temporal_mode=TemporalMode.ROLLING,
        null_hypothesis="rho(lag)=0 within each window",
        alternative_hypothesis="nonzero rolling autocorrelation",
        parameters={"window": w, "lag": k, "min_periods": mp},
        metadata={"n": n},
    )


def bartlett_bands(
    acf_values: np.ndarray | list[float],
    n: int,
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Bartlett ±z·SE significance bands from an ACF series."""
    r = as_float_array(acf_values)
    m = r.size
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = np.empty(m, dtype=np.float64)
    se[0] = 0.0
    for k in range(1, m):
        if k == 1:
            se[k] = 1.0 / np.sqrt(max(n, 1))
        else:
            se[k] = np.sqrt((1.0 + 2.0 * np.sum(r[1:k] ** 2)) / max(n, 1))
    return -z * se, z * se
