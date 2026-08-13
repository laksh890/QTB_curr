"""Mutual information via histogram estimator."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def mutual_information(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    n_bins: int = 16,
    lag: int = 0,
) -> AnalysisResult:
    """Histogram mutual information I(X; Y_{t+lag}) in nats (FULL_SAMPLE)."""
    a = as_float_array(x)
    b = as_float_array(y)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    k = int(lag)
    if k > 0:
        a, b = a[: n - k], b[k:]
    elif k < 0:
        a, b = a[-k:], b[: n + k]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    m = a.size
    bins = max(int(n_bins), 2)
    if m < bins * 2:
        return AnalysisResult(
            method="mutual_information",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="I(X;Y)=0 (independence)",
            alternative_hypothesis="I(X;Y)>0 (statistical dependence)",
            parameters={"n_bins": bins, "lag": k},
        )
    # joint histogram
    c_xy, xedges, yedges = np.histogram2d(a, b, bins=bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    # MI = Σ p_xy log(p_xy / (p_x p_y))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_xy[i, j] <= 0 or p_x[i] <= 0 or p_y[j] <= 0:
                continue
            mi += float(p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j])))
    # normalized MI by min(H(X), H(Y))
    Hx = float(-np.sum(p_x[p_x > 0] * np.log(p_x[p_x > 0])))
    Hy = float(-np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0])))
    denom = min(Hx, Hy)
    nmi = mi / denom if denom > 1e-15 else 0.0
    return AnalysisResult(
        method="mutual_information",
        value=float(mi),
        statistic=float(mi),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="I(X;Y)=0 (independence)",
        alternative_hypothesis="I(X;Y)>0 (statistical dependence)",
        significant=nmi > 0.05,
        parameters={"n_bins": bins, "lag": k},
        metadata={"n": m, "normalized_mi": nmi, "H_x": Hx, "H_y": Hy},
    )
