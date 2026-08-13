"""Regime scenario evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

__all__ = [
    "classify_simple_regimes",
    "run_regime_scenario",
    "evaluate_regime_robustness",
]

# Canonical regime names (labels only — detection is data-driven)
TRENDING = "trending"
MEAN_REVERTING = "mean_reverting"
HIGH_VOL = "high_volatility"
LOW_VOL = "low_volatility"
HIGH_CORR = "high_correlation"
LOW_CORR = "low_correlation"
LOW_LIQUIDITY = "low_liquidity"
REGIME_TRANSITION = "regime_transition"


def classify_simple_regimes(
    returns: Any,
    *,
    vol_window: int = 21,
    trend_window: int = 63,
    high_vol_percentile: float = 0.7,
    low_vol_percentile: float = 0.3,
) -> np.ndarray:
    """Heuristic regime labels from a univariate return series.

    Labels: trending / mean_reverting / high_volatility / low_volatility.
    Transitions are marked where the label changes.
    """
    r = as_returns(returns)
    n = r.size
    labels = np.full(n, MEAN_REVERTING, dtype=object)
    if n == 0:
        return labels

    # Rolling vol
    vol = np.full(n, np.nan)
    vw = max(int(vol_window), 2)
    for i in range(vw - 1, n):
        vol[i] = float(np.std(r[i - vw + 1 : i + 1], ddof=1))
    finite_vol = vol[np.isfinite(vol)]
    if finite_vol.size:
        hi = float(np.quantile(finite_vol, high_vol_percentile))
        lo = float(np.quantile(finite_vol, low_vol_percentile))
    else:
        hi, lo = 0.0, 0.0

    tw = max(int(trend_window), 2)
    for i in range(n):
        if np.isfinite(vol[i]):
            if vol[i] >= hi:
                labels[i] = HIGH_VOL
                continue
            if vol[i] <= lo:
                labels[i] = LOW_VOL
                continue
        if i >= tw - 1:
            window = r[i - tw + 1 : i + 1]
            # Autocorr proxy: sign persistence vs mean reversion
            if window.size > 2:
                ac = float(np.corrcoef(window[:-1], window[1:])[0, 1])
                if np.isfinite(ac) and ac > 0.05:
                    labels[i] = TRENDING
                elif np.isfinite(ac) and ac < -0.05:
                    labels[i] = MEAN_REVERTING

    # Mark transitions
    out = labels.copy()
    for i in range(1, n):
        if labels[i] != labels[i - 1]:
            out[i] = REGIME_TRANSITION
    return out


def run_regime_scenario(
    returns: Any,
    regimes: Any,
    *,
    regime: str | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Evaluate performance within one regime or across all regimes."""
    r = as_returns(returns)
    labs = np.asarray(regimes).reshape(-1)
    n = min(r.size, labs.size)
    r = r[:n]
    labs = np.asarray([str(x) for x in labs[:n].tolist()], dtype=object)

    if regime is not None:
        mask = labs == str(regime)
        window = r[mask]
        return {
            "name": f"regime:{regime}",
            "kind": "regime",
            "regime": str(regime),
            "n_obs": int(window.size),
            "total_return": total_return(window),
            "sharpe": sharpe_ratio(window, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(window),
            "returns": window,
        }

    by_regime: dict[str, Any] = {}
    for lab in sorted(set(labs.tolist())):
        window = r[labs == lab]
        by_regime[lab] = {
            "n_obs": int(window.size),
            "total_return": total_return(window),
            "sharpe": sharpe_ratio(window, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(window),
        }
    return {"name": "regime_all", "kind": "regime", "by_regime": by_regime}


def evaluate_regime_robustness(
    returns: Any,
    regimes: Any,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Score how consistently the strategy performs across regimes."""
    report = run_regime_scenario(returns, regimes, periods_per_year=periods_per_year)
    by = report["by_regime"]
    sharpes = np.array([v["sharpe"] for v in by.values()], dtype=np.float64)
    if sharpes.size == 0:
        score = 0.0
    else:
        score = float(np.mean(sharpes >= 0.0) * (1.0 + np.min(sharpes)))
    return {
        "score": score,
        "n_regimes": len(by),
        "sharpes": {k: float(v["sharpe"]) for k, v in by.items()},
        "by_regime": by,
    }
