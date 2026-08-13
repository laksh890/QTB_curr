"""Market volatility estimators (realized and EWMA)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns

_TRADING_DAYS = 252.0


def realized_volatility(
    returns: Any,
    *,
    window: int | None = None,
    annualize: bool = True,
    trading_days: float = _TRADING_DAYS,
) -> RiskMeasure:
    """Sample standard deviation of returns (optionally rolling window, annualized)."""
    r = as_returns(returns)
    if r.size == 0:
        vol = 0.0
        n = 0
    else:
        if window is not None and window > 0:
            w = min(int(window), r.size)
            sample = r[-w:]
        else:
            sample = r
            w = r.size
        n = int(sample.size)
        vol = float(np.std(sample, ddof=1)) if n > 1 else 0.0
        if annualize:
            vol *= float(np.sqrt(trading_days))
    return RiskMeasure(
        name="realized_volatility",
        value=vol,
        unit="volatility",
        method="sample_std",
        parameters={
            "window": window,
            "annualize": annualize,
            "trading_days": trading_days,
            "n_obs": n,
        },
    )


def ewma_volatility(
    returns: Any,
    *,
    lambda_: float = 0.94,
    annualize: bool = True,
    trading_days: float = _TRADING_DAYS,
    initial_variance: float | None = None,
) -> RiskMeasure:
    """RiskMetrics-style EWMA volatility of the latest observation.

    Uses only past and current returns — no look-ahead.
    """
    r = as_returns(returns)
    lam = float(np.clip(lambda_, 1e-6, 1.0 - 1e-6))
    if r.size == 0:
        return RiskMeasure(
            name="ewma_volatility",
            value=0.0,
            unit="volatility",
            method="ewma",
            parameters={"lambda": lam, "annualize": annualize, "trading_days": trading_days, "n_obs": 0},
        )

    if initial_variance is not None and np.isfinite(initial_variance) and initial_variance >= 0:
        var = float(initial_variance)
    else:
        # Seed with first squared return (causal)
        var = float(r[0] ** 2)

    for t in range(1, r.size):
        var = lam * var + (1.0 - lam) * float(r[t] ** 2)

    vol = float(np.sqrt(max(var, 0.0)))
    if annualize:
        vol *= float(np.sqrt(trading_days))

    return RiskMeasure(
        name="ewma_volatility",
        value=vol,
        unit="volatility",
        method="ewma",
        parameters={
            "lambda": lam,
            "annualize": annualize,
            "trading_days": trading_days,
            "n_obs": int(r.size),
            "last_variance": float(var),
        },
    )
