"""Return metrics: total, CAGR, annualized, daily/monthly, rolling, compounded."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "as_returns",
    "wealth_index",
    "total_return",
    "compounded_return",
    "cagr",
    "annualized_return",
    "annualized_volatility",
    "daily_returns",
    "monthly_returns",
    "rolling_return",
    "summarize_returns",
]


def as_returns(returns: Any) -> np.ndarray:
    """Coerce to 1-D finite float returns (NaN/Inf dropped)."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    return r[np.isfinite(r)]


def wealth_index(returns: Any, *, start: float = 1.0) -> np.ndarray:
    """Cumulative wealth path from simple returns."""
    r = as_returns(returns)
    if r.size == 0:
        return np.array([float(start)], dtype=np.float64)
    return float(start) * np.cumprod(1.0 + r)


def total_return(returns: Any) -> float:
    """Total compounded return over the sample."""
    r = as_returns(returns)
    if r.size == 0:
        return 0.0
    return float(np.prod(1.0 + r) - 1.0)


def compounded_return(returns: Any) -> float:
    """Alias for :func:`total_return`."""
    return total_return(returns)


def cagr(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
    n_periods: int | None = None,
) -> float:
    """Compound annual growth rate from period returns."""
    r = as_returns(returns)
    n = int(n_periods) if n_periods is not None else int(r.size)
    if n <= 0 or r.size == 0:
        return 0.0
    tot = float(np.prod(1.0 + r) - 1.0)
    years = float(n) / max(float(periods_per_year), 1e-12)
    if years <= 0.0:
        return 0.0
    base = 1.0 + tot
    if base <= 0.0:
        return float("nan")
    return float(base ** (1.0 / years) - 1.0)


def annualized_return(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """Arithmetic mean return annualized (``mean * periods_per_year``)."""
    r = as_returns(returns)
    if r.size == 0:
        return 0.0
    return float(np.mean(r) * float(periods_per_year))


def annualized_volatility(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """Sample std of returns annualized."""
    r = as_returns(returns)
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(float(periods_per_year)))


def daily_returns(
    returns: Any,
    *,
    bars_per_day: int = 1,
) -> np.ndarray:
    """Aggregate finer bars into daily compounded returns.

    If ``bars_per_day`` is 1, returns a copy of the input series.
    """
    r = as_returns(returns)
    bpd = max(int(bars_per_day), 1)
    if bpd == 1 or r.size == 0:
        return r.copy()
    n = (r.size // bpd) * bpd
    if n == 0:
        return np.array([], dtype=np.float64)
    blocks = r[:n].reshape(-1, bpd)
    return np.prod(1.0 + blocks, axis=1) - 1.0


def monthly_returns(
    returns: Any,
    *,
    periods_per_month: int = 21,
) -> np.ndarray:
    """Aggregate period returns into monthly compounded returns."""
    return daily_returns(returns, bars_per_day=max(int(periods_per_month), 1))


def rolling_return(
    returns: Any,
    *,
    window: int = 21,
    compounded: bool = True,
) -> np.ndarray:
    """Rolling window return series (NaN-padded for the first ``window-1`` bars)."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    w = max(int(window), 1)
    out = np.full(r.size, np.nan, dtype=np.float64)
    if r.size < w:
        return out
    if compounded:
        # log-sum-exp of (1+r) via cumprod ratios
        wealth = np.cumprod(1.0 + np.nan_to_num(r, nan=0.0))
        lagged = np.empty_like(wealth)
        lagged[:w] = 1.0
        lagged[w:] = wealth[:-w]
        out[w - 1 :] = wealth[w - 1 :] / np.maximum(lagged[w - 1 :], 1e-12) - 1.0
    else:
        csum = np.cumsum(np.nan_to_num(r, nan=0.0))
        out[w - 1] = csum[w - 1]
        out[w:] = csum[w:] - csum[:-w]
    finite = np.isfinite(r)
    # Invalidate windows that contain non-finite inputs
    if not np.all(finite):
        bad = ~finite
        # mark any window covering a bad bar
        for i in range(w - 1, r.size):
            if np.any(bad[i - w + 1 : i + 1]):
                out[i] = np.nan
    return out


def summarize_returns(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Compact return summary dict."""
    r = as_returns(returns)
    return {
        "n_obs": float(r.size),
        "total_return": total_return(r),
        "compounded_return": compounded_return(r),
        "cagr": cagr(r, periods_per_year=periods_per_year),
        "annualized_return": annualized_return(r, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(r, periods_per_year=periods_per_year),
        "mean": float(np.mean(r)) if r.size else 0.0,
        "std": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
    }
