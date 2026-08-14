"""Regime scenario evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

__all__ = [
    "classify_simple_regimes",
    "evaluate_regime_robustness",
    "run_regime_scenario",
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

    Vectorized for large intraday series (e.g. multi-million crypto bars).
    """
    import pandas as pd

    r = as_returns(returns)
    n = r.size
    labels = np.full(n, MEAN_REVERTING, dtype=object)
    if n == 0:
        return labels

    vw = max(int(vol_window), 2)
    tw = max(int(trend_window), 2)
    s = pd.Series(r)
    vol = s.rolling(vw, min_periods=vw).std(ddof=1).to_numpy(dtype=np.float64)
    finite_vol = vol[np.isfinite(vol)]
    if finite_vol.size:
        hi = float(np.quantile(finite_vol, high_vol_percentile))
        lo = float(np.quantile(finite_vol, low_vol_percentile))
    else:
        hi, lo = 0.0, 0.0

    # Autocorr proxy via rolling corr of r_t vs r_{t-1}
    ac = s.rolling(tw, min_periods=tw).corr(s.shift(1)).to_numpy(dtype=np.float64)

    high_vol = np.isfinite(vol) & (vol >= hi)
    low_vol = np.isfinite(vol) & (vol <= lo) & ~high_vol
    trending = np.isfinite(ac) & (ac > 0.05) & ~high_vol & ~low_vol
    mean_rev = np.isfinite(ac) & (ac < -0.05) & ~high_vol & ~low_vol

    labels[mean_rev] = MEAN_REVERTING
    labels[trending] = TRENDING
    labels[low_vol] = LOW_VOL
    labels[high_vol] = HIGH_VOL

    out = labels.copy()
    changed = np.empty(n, dtype=bool)
    changed[0] = False
    changed[1:] = labels[1:] != labels[:-1]
    out[changed] = REGIME_TRANSITION
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
    labs = labs[:n].astype(str)

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
