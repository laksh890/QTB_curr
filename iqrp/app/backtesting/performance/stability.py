"""Rolling stability analytics."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, rolling_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio
from iqrp.app.backtesting.performance.trade_metrics import turnover

__all__ = [
    "rolling_costs",
    "rolling_drawdown",
    "rolling_ic",
    "rolling_return_series",
    "rolling_sharpe",
    "rolling_turnover",
    "rolling_volatility",
    "stability_report",
]


def _rolling_apply(
    values: np.ndarray,
    window: int,
    fn,
) -> np.ndarray:
    w = max(int(window), 1)
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for i in range(w - 1, values.shape[0]):
        sl = values[i - w + 1 : i + 1]
        if np.all(np.isfinite(sl)):
            out[i] = float(fn(sl))
    return out


def rolling_sharpe(
    returns: Any,
    *,
    window: int = 63,
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """Rolling annualized Sharpe."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)

    def _sh(sl: np.ndarray) -> float:
        return sharpe_ratio(sl, risk_free=risk_free, periods_per_year=periods_per_year)

    return _rolling_apply(r, window, _sh)


def rolling_return_series(
    returns: Any,
    *,
    window: int = 21,
    compounded: bool = True,
) -> np.ndarray:
    """Rolling compounded (or summed) return."""
    return rolling_return(returns, window=window, compounded=compounded)


def rolling_drawdown(
    returns: Any,
    *,
    window: int = 63,
) -> np.ndarray:
    """Rolling maximum drawdown over trailing window."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    return _rolling_apply(r, window, max_drawdown)


def rolling_volatility(
    returns: Any,
    *,
    window: int = 63,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """Rolling annualized volatility."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)

    def _vol(sl: np.ndarray) -> float:
        if sl.size < 2:
            return 0.0
        return float(np.std(sl, ddof=1) * np.sqrt(float(periods_per_year)))

    return _rolling_apply(r, window, _vol)


def rolling_ic(
    forecasts: Any,
    realized: Any,
    *,
    window: int = 63,
) -> np.ndarray:
    """Rolling Spearman rank IC between forecasts and realized returns."""
    f = np.asarray(forecasts, dtype=np.float64).reshape(-1)
    y = np.asarray(realized, dtype=np.float64).reshape(-1)
    n = min(f.size, y.size)
    f = f[:n]
    y = y[:n]
    w = max(int(window), 2)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(w - 1, n):
        a = f[i - w + 1 : i + 1]
        b = y[i - w + 1 : i + 1]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(np.sum(mask)) < 3:
            continue
        aa = a[mask]
        bb = b[mask]
        ra = aa.argsort().argsort().astype(np.float64)
        rb = bb.argsort().argsort().astype(np.float64)
        if float(np.std(ra)) < 1e-15 or float(np.std(rb)) < 1e-15:
            out[i] = 0.0
        else:
            out[i] = float(np.corrcoef(ra, rb)[0, 1])
    return out


def rolling_turnover(
    positions: Any,
    *,
    window: int = 21,
) -> np.ndarray:
    """Rolling mean absolute position change."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim == 1:
        deltas = np.abs(np.diff(pos, prepend=pos[:1]))
    elif pos.ndim == 2:
        deltas = np.sum(np.abs(np.diff(pos, axis=0, prepend=pos[:1])), axis=1)
    else:
        raise ValueError("positions must be 1-D or 2-D")
    return _rolling_apply(deltas, window, lambda sl: float(np.mean(sl)))


def rolling_costs(
    costs: Any,
    *,
    window: int = 21,
) -> np.ndarray:
    """Rolling mean absolute cost per period."""
    c = np.asarray(costs, dtype=np.float64).reshape(-1)
    return _rolling_apply(c, window, lambda sl: float(np.mean(np.abs(sl))))


def stability_report(
    returns: Any,
    *,
    window: int = 63,
    positions: Any | None = None,
    costs: Any | None = None,
    forecasts: Any | None = None,
    realized: Any | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Bundle of rolling stability series and summary statistics."""
    r = as_returns(returns)
    sharpes = rolling_sharpe(r, window=window, periods_per_year=periods_per_year)
    rets = rolling_return_series(r, window=window)
    dds = rolling_drawdown(r, window=window)
    vols = rolling_volatility(r, window=window, periods_per_year=periods_per_year)

    def _finite_stats(x: np.ndarray) -> dict[str, float]:
        f = x[np.isfinite(x)]
        if f.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(f)),
            "std": float(np.std(f, ddof=1)) if f.size > 1 else 0.0,
            "min": float(np.min(f)),
            "max": float(np.max(f)),
        }

    report: dict[str, Any] = {
        "window": int(window),
        "rolling_sharpe": sharpes,
        "rolling_return": rets,
        "rolling_drawdown": dds,
        "rolling_volatility": vols,
        "sharpe_stability": _finite_stats(sharpes),
        "return_stability": _finite_stats(rets),
        "drawdown_stability": _finite_stats(dds),
        "volatility_stability": _finite_stats(vols),
    }
    if positions is not None:
        to = rolling_turnover(positions, window=window)
        report["rolling_turnover"] = to
        report["turnover_stability"] = _finite_stats(to)
        report["mean_turnover"] = turnover(positions)
    if costs is not None:
        rc = rolling_costs(costs, window=window)
        report["rolling_costs"] = rc
        report["cost_stability"] = _finite_stats(rc)
    if forecasts is not None and realized is not None:
        ic = rolling_ic(forecasts, realized, window=window)
        report["rolling_ic"] = ic
        report["ic_stability"] = _finite_stats(ic)
    return report
