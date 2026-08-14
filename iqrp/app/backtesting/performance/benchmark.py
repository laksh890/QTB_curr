"""Benchmark-relative performance analysis."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.backtesting.performance.returns import (
    annualized_return,
    as_returns,
    cagr,
    total_return,
)
from iqrp.app.backtesting.performance.risk_adjusted import (
    capture_ratios,
    information_ratio,
    sharpe_ratio,
)

BenchmarkKind = Literal["buyhold", "market", "risk_free", "custom", "strategy"]

__all__ = [
    "active_returns",
    "buy_and_hold_returns",
    "compare_to_benchmark",
    "relative_performance",
    "risk_free_returns",
]


def buy_and_hold_returns(
    asset_returns: Any,
    *,
    weights: Any | None = None,
) -> np.ndarray:
    """Buy-and-hold portfolio returns from asset returns.

    With equal or supplied initial weights held constant (no rebalance).
    For multi-asset, uses beginning-of-sample weights on each period's returns
    (static mix approximation).
    """
    r = np.asarray(asset_returns, dtype=np.float64)
    if r.ndim == 1:
        return as_returns(r)
    n_assets = r.shape[1]
    if weights is None:
        w = np.full(n_assets, 1.0 / max(n_assets, 1))
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != n_assets:
            raise ValueError("weights length must match assets")
        s = float(np.sum(np.abs(w)))
        if s > 1e-15:
            w = w / s
    return r @ w


def risk_free_returns(
    n: int,
    *,
    rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """Constant per-period risk-free return series from annualized rate."""
    per = float(rate) / max(float(periods_per_year), 1e-12)
    return np.full(max(int(n), 0), per, dtype=np.float64)


def active_returns(returns: Any, benchmark: Any) -> np.ndarray:
    """Strategy minus benchmark (aligned length)."""
    r = as_returns(returns)
    b = as_returns(benchmark)
    n = min(r.size, b.size)
    return r[:n] - b[:n]


def relative_performance(
    returns: Any,
    benchmark: Any,
    *,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Relative stats vs a benchmark series."""
    r = as_returns(returns)
    b = as_returns(benchmark)
    n = min(r.size, b.size)
    r = r[:n]
    b = b[:n]
    active = r - b
    out = {
        "total_return": total_return(r),
        "benchmark_total_return": total_return(b),
        "active_total_return": total_return(active),
        "cagr": cagr(r, periods_per_year=periods_per_year),
        "benchmark_cagr": cagr(b, periods_per_year=periods_per_year),
        "annualized_return": annualized_return(r, periods_per_year=periods_per_year),
        "benchmark_annualized_return": annualized_return(b, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free=risk_free, periods_per_year=periods_per_year),
        "benchmark_sharpe": sharpe_ratio(b, risk_free=risk_free, periods_per_year=periods_per_year),
        "information_ratio": information_ratio(r, b, periods_per_year=periods_per_year),
        "tracking_error": (
            float(np.std(active, ddof=1) * np.sqrt(periods_per_year)) if active.size > 1 else 0.0
        ),
    }
    out.update(capture_ratios(r, b))
    return out


def compare_to_benchmark(
    returns: Any,
    *,
    kind: BenchmarkKind = "market",
    benchmark: Any | None = None,
    asset_returns: Any | None = None,
    buyhold_weights: Any | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Compare strategy returns to a named benchmark type.

    Parameters
    ----------
    kind :
        ``buyhold`` — static mix of ``asset_returns``;
        ``market`` / ``custom`` / ``strategy`` — use ``benchmark`` series;
        ``risk_free`` — constant RF series.
    """
    r = as_returns(returns)
    k = str(kind).lower()
    if k == "buyhold":
        if asset_returns is None:
            raise ValueError("asset_returns required for buyhold benchmark")
        bench = buy_and_hold_returns(asset_returns, weights=buyhold_weights)
    elif k == "risk_free":
        bench = risk_free_returns(r.size, rate=risk_free_rate, periods_per_year=periods_per_year)
    else:
        if benchmark is None:
            raise ValueError(f"benchmark series required for kind={kind!r}")
        bench = as_returns(benchmark)

    stats = relative_performance(
        r,
        bench,
        periods_per_year=periods_per_year,
        risk_free=risk_free_rate / max(periods_per_year, 1e-12),
    )
    return {"kind": k, "benchmark_returns": bench, **stats}
