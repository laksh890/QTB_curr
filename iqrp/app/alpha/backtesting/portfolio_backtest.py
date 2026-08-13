"""Multi-asset / weight-path portfolio backtests.

Look-ahead prevention
---------------------
``weights[t]`` must be decided using information available at ``t``.
``returns[t]`` must be the subsequent holding-period return. When
``returns_are_forward=False``, returns are lagged by one bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _ann_sharpe(r: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def portfolio_backtest(
    weights: Any,
    returns: Any,
    *,
    cost_bps: float = 0.0,
    returns_are_forward: bool = True,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Backtest a weight path against asset returns.

    Parameters
    ----------
    weights:
        ``(T,)`` single-asset exposure or ``(T, N)`` weight matrix.
    returns:
        Same shape as ``weights`` (or broadcastable ``(T,)`` / ``(T, N)``).
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    t = min(w.shape[0], r.shape[0])
    n = min(w.shape[1], r.shape[1])
    w = w[:t, :n].copy()
    r = r[:t, :n].copy()

    if not returns_are_forward:
        if t < 2:
            w = w[:0]
            r = r[:0]
            t = 0
        else:
            w = w[:-1]
            r = r[1:]
            t = w.shape[0]

    if t == 0:
        return {
            "gross_returns": np.asarray([], dtype=np.float64),
            "net_returns": np.asarray([], dtype=np.float64),
            "turnover": np.asarray([], dtype=np.float64),
            "gross_sharpe": float("nan"),
            "net_sharpe": float("nan"),
            "n": 0,
            "look_ahead_guard": "returns_are_forward" if returns_are_forward else "shifted_returns",
        }

    gross = np.nansum(w * r, axis=1)
    # turnover: half L1 change (portfolio convention) * 2 for full book = L1/2 * 2
    dw = np.zeros(t, dtype=np.float64)
    dw[0] = 0.5 * float(np.nansum(np.abs(w[0])))
    if t > 1:
        dw[1:] = 0.5 * np.nansum(np.abs(np.diff(w, axis=0)), axis=1)
    costs = dw * (float(cost_bps) / 1e4)
    net = gross - costs

    return {
        "gross_returns": gross,
        "net_returns": net,
        "turnover": dw,
        "weights": w,
        "gross_sharpe": _ann_sharpe(gross, periods_per_year=periods_per_year),
        "net_sharpe": _ann_sharpe(net, periods_per_year=periods_per_year),
        "gross_mean": float(np.nanmean(gross)),
        "net_mean": float(np.nanmean(net)),
        "total_cost": float(np.nansum(costs)),
        "avg_turnover": float(np.nanmean(dw)),
        "n": int(t),
        "n_assets": int(n),
        "cost_bps": float(cost_bps),
        "look_ahead_guard": "returns_are_forward" if returns_are_forward else "shifted_returns",
        "cumulative_net": float(np.nansum(net)),
    }
