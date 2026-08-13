"""Partial autocorrelation via Durbin-Levinson recursion."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.autocorrelation.acf import acf as compute_acf


def pacf(
    x: np.ndarray | list[float],
    *,
    nlags: int | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    """PACF via Durbin-Levinson (FULL_SAMPLE).

    Returns PACF values for lags 0..nlags where pacf[0]=1.
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    if n < 4:
        return AnalysisResult(
            method="pacf",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="phi_{kk}=0 (no additional AR dependence at lag k)",
            alternative_hypothesis="nonzero partial autocorrelation at some lag",
            parameters={"nlags": nlags, "alpha": alpha},
        )
    max_lag = int(nlags) if nlags is not None else min(n - 1, int(np.floor(10 * np.log10(n))))
    max_lag = int(np.clip(max_lag, 1, n - 1))

    acf_res = compute_acf(finite, nlags=max_lag)
    if isinstance(acf_res.value, str):
        return AnalysisResult(
            method="pacf",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="phi_{kk}=0 (no additional AR dependence at lag k)",
            alternative_hypothesis="nonzero partial autocorrelation at some lag",
            parameters={"nlags": max_lag, "alpha": alpha},
        )
    r = np.asarray(acf_res.value, dtype=np.float64)
    pacf_vals = _durbin_levinson(r, max_lag)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = 1.0 / np.sqrt(n)
    band = z * se
    # test lag-1 PACF against white-noise bands
    sig_lags = [int(k) for k in range(1, max_lag + 1) if abs(pacf_vals[k]) > band]
    # approximate joint p via max |PACF|
    max_abs = float(np.max(np.abs(pacf_vals[1:]))) if max_lag >= 1 else 0.0
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(max_abs * np.sqrt(n))))

    return AnalysisResult(
        method="pacf",
        value=pacf_vals,
        statistic=max_abs,
        pvalue=float(np.clip(pvalue, 0.0, 1.0)),
        confidence=1.0 - alpha,
        confidence_interval=(-band, band),
        null_hypothesis="phi_{kk}=0 (no additional AR dependence at lag k)",
        alternative_hypothesis="nonzero partial autocorrelation at some lag",
        significant=len(sig_lags) > 0,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"nlags": max_lag, "alpha": alpha},
        metadata={"lags": list(range(max_lag + 1)), "significant_lags": sig_lags, "n": n},
    )


def _durbin_levinson(acf_vals: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute PACF from ACF using Durbin-Levinson recursion."""
    pacf_out = np.zeros(max_lag + 1, dtype=np.float64)
    pacf_out[0] = 1.0
    if max_lag < 1:
        return pacf_out
    phi = np.zeros((max_lag + 1, max_lag + 1), dtype=np.float64)
    # phi[k, j] = AR(k) coefficient for lag j (1-indexed j)
    v = np.zeros(max_lag + 1, dtype=np.float64)
    phi[1, 1] = acf_vals[1]
    pacf_out[1] = phi[1, 1]
    v[1] = 1.0 - phi[1, 1] ** 2
    for k in range(2, max_lag + 1):
        if abs(v[k - 1]) < 1e-18:
            pacf_out[k:] = 0.0
            break
        num = acf_vals[k] - np.sum(phi[k - 1, 1:k] * acf_vals[1:k][::-1])
        phi[k, k] = num / v[k - 1]
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf_out[k] = phi[k, k]
        v[k] = v[k - 1] * (1.0 - phi[k, k] ** 2)
        # numerical guard
        pacf_out[k] = float(np.clip(pacf_out[k], -1.0, 1.0))
    return pacf_out
