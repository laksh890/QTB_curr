"""Cross-correlation and lead-lag analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def ccf(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    nlags: int | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Cross-correlation function for lags -nlags..+nlags (FULL_SAMPLE).

    Positive lag means y leads x (x correlates with future y? convention:
    ccf(k) = Corr(x_t, y_{t+k}) so positive k ⇒ y leads x).
    """
    a = as_float_array(x)
    b = as_float_array(y)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 5:
        return AnalysisResult(
            method="ccf",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="rho_xy(k)=0 for all k (independence at all lags)",
            alternative_hypothesis="nonzero cross-correlation at some lag",
            parameters={"nlags": nlags, "alpha": alpha},
        )
    max_lag = int(nlags) if nlags is not None else min(n // 4, int(np.floor(10 * np.log10(n))))
    max_lag = int(np.clip(max_lag, 1, n - 2))

    a = a - np.mean(a)
    b = b - np.mean(b)
    sa = float(np.std(a, ddof=0))
    sb = float(np.std(b, ddof=0))
    if sa < 1e-18 or sb < 1e-18:
        return AnalysisResult(
            method="ccf",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="rho_xy(k)=0 for all k (independence at all lags)",
            alternative_hypothesis="nonzero cross-correlation at some lag",
            metadata={"reason": "zero_variance"},
            parameters={"nlags": max_lag, "alpha": alpha},
        )

    lags = np.arange(-max_lag, max_lag + 1)
    vals = np.empty(lags.size, dtype=np.float64)
    for i, k in enumerate(lags):
        vals[i] = _cross_corr_at_lag(a, b, int(k), sa, sb, n)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    band = z / np.sqrt(n)
    max_abs = float(np.max(np.abs(vals)))
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(max_abs * np.sqrt(n))))
    best = int(lags[int(np.argmax(np.abs(vals)))])

    return AnalysisResult(
        method="ccf",
        value=vals,
        statistic=max_abs,
        pvalue=float(np.clip(pvalue, 0.0, 1.0)),
        confidence=1.0 - alpha,
        confidence_interval=(-band, band),
        null_hypothesis="rho_xy(k)=0 for all k (independence at all lags)",
        alternative_hypothesis="nonzero cross-correlation at some lag",
        significant=max_abs > band,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"nlags": max_lag, "alpha": alpha},
        metadata={"lags": lags.tolist(), "best_lag": best, "n": n, "band": band},
    )


def lead_lag(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    max_lag: int | None = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    """Identify the lag of maximum |CCF| between x and y (FULL_SAMPLE)."""
    res = ccf(x, y, nlags=max_lag, alpha=alpha)
    if isinstance(res.value, str):
        return AnalysisResult(
            method="lead_lag",
            value="insufficient_data",
            statistic=np.nan,
            pvalue=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no lead-lag relationship (max |CCF| insignificant)",
            alternative_hypothesis="one series leads the other at some lag",
            parameters={"max_lag": max_lag, "alpha": alpha},
        )
    lags = np.asarray(res.metadata.get("lags", []), dtype=int)
    vals = np.asarray(res.value, dtype=np.float64)
    idx = int(np.argmax(np.abs(vals)))
    best_lag = int(lags[idx]) if lags.size else 0
    best_corr = float(vals[idx])
    band = float(res.metadata.get("band", 0.0))
    if best_lag > 0:
        relation = "y_leads_x"
    elif best_lag < 0:
        relation = "x_leads_y"
    else:
        relation = "contemporaneous"
    return AnalysisResult(
        method="lead_lag",
        value={"lag": best_lag, "correlation": best_corr, "relation": relation},
        statistic=best_corr,
        pvalue=res.pvalue,
        confidence=1.0 - alpha,
        null_hypothesis="no lead-lag relationship (max |CCF| insignificant)",
        alternative_hypothesis="one series leads the other at some lag",
        significant=abs(best_corr) > band,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"max_lag": res.parameters.get("nlags"), "alpha": alpha},
        metadata={"ccf": vals.tolist(), "lags": lags.tolist(), "band": band},
    )


def _cross_corr_at_lag(a: np.ndarray, b: np.ndarray, k: int, sa: float, sb: float, n: int) -> float:
    if k >= 0:
        # Corr(x_t, y_{t+k})
        num = float(np.dot(a[: n - k], b[k:])) / n
    else:
        kk = -k
        num = float(np.dot(a[kk:], b[: n - kk])) / n
    return num / (sa * sb)
