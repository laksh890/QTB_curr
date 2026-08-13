"""Seasonality diagnostics for signal / return association.

CRITICAL:
- Seasonal patterns in-sample are fragile; not alpha by themselves.
- Statistical significance alone ≠ alpha.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.alpha.research.decay import forward_returns
from iqrp.app.alpha.research.information_coefficient import compute_ic
from iqrp.app.features.research._numeric import safe_nanmean


def _period_labels(n: int, period: int) -> np.ndarray:
    return np.arange(n) % period


def analyze_seasonality(
    signal: np.ndarray,
    returns: np.ndarray,
    *,
    period: int = 5,
    horizon: int = 1,
) -> dict[str, Any]:
    """IC within calendar-like buckets ``t % period``.

    Without timestamps, uses modular index buckets as a proxy seasonality grid
    (e.g. period=5 ≈ weekday if daily bars start Monday).
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    x = np.asarray(signal, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if len(x) != len(r):
        raise ValueError("length mismatch")
    fwd = forward_returns(r, horizon)
    labels = _period_labels(len(x), period)
    by_bucket: dict[int, float] = {}
    counts: dict[int, int] = {}
    for b in range(period):
        mask = labels == b
        xs, ys = x[mask], fwd[mask]
        n_fin = int((np.isfinite(xs) & np.isfinite(ys)).sum())
        counts[b] = n_fin
        by_bucket[b] = compute_ic(xs, ys) if n_fin >= 10 else float("nan")
    vals = np.asarray(list(by_bucket.values()), dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    dispersion = float(np.std(finite)) if finite.size else float("nan")
    return {
        "period": period,
        "horizon": horizon,
        "ic_by_bucket": by_bucket,
        "n_by_bucket": counts,
        "mean_abs_ic": safe_nanmean(np.abs(vals)),
        "ic_dispersion": dispersion,
        "disclaimer": (
            "Seasonality diagnostics are exploratory. "
            "Statistical significance alone ≠ alpha."
        ),
    }


def month_of_year_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    months: np.ndarray,
) -> dict[str, Any]:
    """IC by calendar month (1-12) when month labels are provided."""
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(forward_returns, dtype=np.float64)
    m = np.asarray(months)
    if not (len(x) == len(y) == len(m)):
        raise ValueError("length mismatch")
    out: dict[int, float] = {}
    for month in range(1, 13):
        mask = m == month
        if mask.sum() < 10:
            out[month] = float("nan")
            continue
        out[month] = compute_ic(x[mask], y[mask])
    return {
        "ic_by_month": out,
        "disclaimer": "Calendar seasonality ≠ alpha.",
    }
