"""Beta versus a benchmark return series."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns


def beta(
    returns: Any,
    benchmark: Any,
    *,
    window: int | None = None,
) -> RiskMeasure:
    """OLS beta of asset returns vs benchmark (causal window on trailing data)."""
    a = as_returns(returns)
    b = as_returns(benchmark)
    n = int(min(a.size, b.size))
    if n == 0:
        return RiskMeasure(
            name="beta",
            value=0.0,
            unit="ratio",
            method="ols",
            parameters={"window": window, "n_obs": 0},
        )

    a = a[-n:]
    b = b[-n:]
    if window is not None and window > 0:
        w = min(int(window), n)
        a = a[-w:]
        b = b[-w:]
        n = w

    if n < 2:
        beta_val = 0.0
        r2 = 0.0
    else:
        b_var = float(np.var(b, ddof=1))
        if b_var <= 0.0 or not np.isfinite(b_var):
            beta_val = 0.0
            r2 = 0.0
        else:
            cov = float(np.cov(a, b, ddof=1)[0, 1])
            beta_val = cov / b_var
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 else 0.0
            r2 = float(corr**2) if np.isfinite(corr) else 0.0

    return RiskMeasure(
        name="beta",
        value=float(beta_val) if np.isfinite(beta_val) else 0.0,
        unit="ratio",
        method="ols",
        parameters={"window": window, "n_obs": n},
        metadata={"r_squared": r2},
    )


def tracking_error(
    returns: Any,
    benchmark: Any,
    *,
    window: int | None = None,
    annualize: bool = True,
    periods_per_year: float = 252.0,
) -> RiskMeasure:
    """Tracking error: std of active returns vs benchmark."""
    a = as_returns(returns)
    b = as_returns(benchmark)
    n = int(min(a.size, b.size))
    if n == 0:
        return RiskMeasure(
            name="tracking_error",
            value=0.0,
            unit="volatility",
            method="active_std",
            parameters={"window": window, "n_obs": 0, "annualize": annualize},
        )
    a = a[-n:]
    b = b[-n:]
    if window is not None and window > 0:
        w = min(int(window), n)
        a = a[-w:]
        b = b[-w:]
        n = w
    active = a - b
    if n < 2:
        te = 0.0
    else:
        te = float(np.std(active, ddof=1))
        if annualize:
            te *= float(np.sqrt(max(periods_per_year, 1.0)))
    return RiskMeasure(
        name="tracking_error",
        value=te if np.isfinite(te) else 0.0,
        unit="volatility",
        method="active_std",
        parameters={
            "window": window,
            "n_obs": n,
            "annualize": bool(annualize),
            "periods_per_year": float(periods_per_year),
        },
    )
