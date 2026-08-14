"""Risk-adjusted performance ratios."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, cagr

__all__ = [
    "calmar_ratio",
    "capture_ratios",
    "downside_capture",
    "information_ratio",
    "omega_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize_risk_adjusted",
    "upside_capture",
]


def sharpe_ratio(
    returns: Any,
    *,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized Sharpe ratio vs constant per-period risk-free rate."""
    r = as_returns(returns)
    if r.size < 2:
        return 0.0
    excess = r - float(risk_free)
    sd = float(np.std(excess, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(float(periods_per_year)))


def sortino_ratio(
    returns: Any,
    *,
    mar: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized Sortino ratio vs minimum acceptable return (MAR)."""
    r = as_returns(returns)
    if r.size < 2:
        return 0.0
    downside = np.minimum(r - float(mar), 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd < 1e-15:
        return 0.0
    return float((np.mean(r) - float(mar)) / dd * np.sqrt(float(periods_per_year)))


def calmar_ratio(
    returns: Any,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """CAGR divided by maximum drawdown."""
    r = as_returns(returns)
    mdd = max_drawdown(r)
    if mdd < 1e-15:
        return 0.0
    return float(cagr(r, periods_per_year=periods_per_year) / mdd)


def omega_ratio(
    returns: Any,
    *,
    threshold: float = 0.0,
) -> float:
    """Omega ratio: gains above threshold / losses below threshold."""
    r = as_returns(returns)
    if r.size == 0:
        return 0.0
    gains = np.maximum(r - float(threshold), 0.0)
    losses = np.maximum(float(threshold) - r, 0.0)
    loss_sum = float(np.sum(losses))
    if loss_sum < 1e-15:
        return float("inf") if float(np.sum(gains)) > 0 else 0.0
    return float(np.sum(gains) / loss_sum)


def information_ratio(
    returns: Any,
    benchmark: Any,
    *,
    periods_per_year: float = 252.0,
) -> float:
    """Annualized information ratio of active returns vs benchmark."""
    r = as_returns(returns)
    b = as_returns(benchmark)
    n = min(r.size, b.size)
    if n < 2:
        return 0.0
    active = r[:n] - b[:n]
    te = float(np.std(active, ddof=1))
    if te < 1e-15:
        return 0.0
    return float(np.mean(active) / te * np.sqrt(float(periods_per_year)))


def upside_capture(
    returns: Any,
    benchmark: Any,
) -> float:
    """Upside capture: strategy return / bench return when bench > 0."""
    r = as_returns(returns)
    b = as_returns(benchmark)
    n = min(r.size, b.size)
    if n == 0:
        return 0.0
    mask = b[:n] > 0.0
    if not np.any(mask):
        return 0.0
    br = float(np.prod(1.0 + b[:n][mask]) - 1.0)
    if abs(br) < 1e-15:
        return 0.0
    sr = float(np.prod(1.0 + r[:n][mask]) - 1.0)
    return float(sr / br)


def downside_capture(
    returns: Any,
    benchmark: Any,
) -> float:
    """Downside capture: strategy return / bench return when bench < 0."""
    r = as_returns(returns)
    b = as_returns(benchmark)
    n = min(r.size, b.size)
    if n == 0:
        return 0.0
    mask = b[:n] < 0.0
    if not np.any(mask):
        return 0.0
    br = float(np.prod(1.0 + b[:n][mask]) - 1.0)
    if abs(br) < 1e-15:
        return 0.0
    sr = float(np.prod(1.0 + r[:n][mask]) - 1.0)
    return float(sr / br)


def capture_ratios(returns: Any, benchmark: Any) -> dict[str, float]:
    """Both upside and downside capture."""
    return {
        "upside_capture": upside_capture(returns, benchmark),
        "downside_capture": downside_capture(returns, benchmark),
    }


def summarize_risk_adjusted(
    returns: Any,
    *,
    benchmark: Any | None = None,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Bundle of common risk-adjusted metrics."""
    out: dict[str, float] = {
        "sharpe": sharpe_ratio(returns, risk_free=risk_free, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, mar=risk_free, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(returns, periods_per_year=periods_per_year),
        "omega": omega_ratio(returns, threshold=risk_free),
    }
    if benchmark is not None:
        out["information_ratio"] = information_ratio(
            returns, benchmark, periods_per_year=periods_per_year
        )
        out.update(capture_ratios(returns, benchmark))
    return out
