"""Hurst exponent via rescaled range (R/S) analysis."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def hurst_exponent(
    x: np.ndarray | list[float],
    *,
    min_window: int = 8,
    max_window: int | None = None,
) -> AnalysisResult:
    """R/S Hurst exponent estimate.

    Statistical descriptor only — not a guaranteed predictive signal.
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    if n < max(min_window * 2, 16):
        return AnalysisResult(
            method="hurst_exponent",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="H=0.5 (random walk / uncorrelated increments)",
            alternative_hypothesis="H≠0.5 (long-memory or mean-reversion)",
            parameters={"min_window": min_window, "max_window": max_window},
        )
    max_w = int(max_window) if max_window is not None else n // 2
    max_w = int(np.clip(max_w, min_window + 1, n // 2))
    # geometrically spaced windows
    windows = np.unique(
        np.geomspace(min_window, max_w, num=min(20, max_w - min_window + 1)).astype(int)
    )
    windows = windows[windows >= min_window]
    rs_vals: list[float] = []
    used: list[int] = []
    for w in windows:
        rs = _mean_rs(finite, int(w))
        if rs > 0:
            rs_vals.append(rs)
            used.append(int(w))
    if len(used) < 2:
        return AnalysisResult(
            method="hurst_exponent",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="H=0.5 (random walk / uncorrelated increments)",
            alternative_hypothesis="H≠0.5 (long-memory or mean-reversion)",
            parameters={"min_window": min_window, "max_window": max_w},
        )
    log_w = np.log(np.asarray(used, dtype=np.float64))
    log_rs = np.log(np.asarray(rs_vals, dtype=np.float64))
    # H = slope of log(R/S) vs log(w)
    A = np.column_stack([np.ones(log_w.size), log_w])
    beta, *_ = np.linalg.lstsq(A, log_rs, rcond=None)
    H = float(beta[1])
    H = float(np.clip(H, 0.0, 1.0))
    return AnalysisResult(
        method="hurst_exponent",
        value=H,
        statistic=H,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="H=0.5 (random walk / uncorrelated increments)",
        alternative_hypothesis="H≠0.5 (long-memory or mean-reversion)",
        significant=abs(H - 0.5) > 0.1,
        parameters={"min_window": min_window, "max_window": max_w},
        metadata={"windows": used, "rs": rs_vals, "n": n},
    )


def _mean_rs(y: np.ndarray, w: int) -> float:
    n = y.size
    n_seg = n // w
    if n_seg < 1:
        return 0.0
    vals = []
    for i in range(n_seg):
        seg = y[i * w : (i + 1) * w]
        mu = np.mean(seg)
        dev = np.cumsum(seg - mu)
        R = float(np.max(dev) - np.min(dev))
        S = float(np.std(seg, ddof=1)) if w > 1 else 0.0
        if S > 1e-15:
            vals.append(R / S)
    return float(np.mean(vals)) if vals else 0.0
