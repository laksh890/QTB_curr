"""Conditional Value-at-Risk (Expected Shortfall) estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from iqrp.app.risk.base import RiskMeasure, as_returns
from iqrp.app.risk.tail.var import _normalize_confidence, _scale_horizon


def historical_cvar(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
) -> RiskMeasure:
    """Historical CVaR / Expected Shortfall (positive loss)."""
    r = as_returns(returns)
    conf = _normalize_confidence(confidence)
    alpha = 1.0 - conf
    if r.size == 0:
        value = 0.0
    else:
        q = float(np.quantile(r, alpha))
        tail = r[r <= q]
        if tail.size == 0:
            es = max(-q, 0.0)
        else:
            es = float(max(-np.mean(tail), 0.0))
        value = _scale_horizon(es, horizon)
    return RiskMeasure(
        name="cvar",
        value=value,
        unit="return",
        confidence=conf,
        horizon=int(horizon),
        method="historical",
        parameters={"n_obs": int(r.size), "alpha": alpha},
    )


def parametric_cvar(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
) -> RiskMeasure:
    """Gaussian parametric CVaR."""
    r = as_returns(returns)
    conf = _normalize_confidence(confidence)
    alpha = 1.0 - conf
    if r.size == 0:
        mu, sigma, value = 0.0, 0.0, 0.0
    else:
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        z = float(stats.norm.ppf(alpha))
        # ES for normal: -mu + sigma * phi(z) / alpha  (loss form for P&L = r)
        dens = float(stats.norm.pdf(z))
        es = -mu + max(sigma, 0.0) * dens / max(alpha, 1e-12)
        value = _scale_horizon(max(es, 0.0), horizon)
    return RiskMeasure(
        name="cvar",
        value=value,
        unit="return",
        confidence=conf,
        horizon=int(horizon),
        method="parametric",
        parameters={"mu": mu, "sigma": sigma, "n_obs": int(r.size), "alpha": alpha},
    )


def monte_carlo_cvar(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 5000,
    seed: int = 42,
) -> RiskMeasure:
    """Monte Carlo CVaR under i.i.d. normal fitted to historical returns."""
    r = as_returns(returns)
    conf = _normalize_confidence(confidence)
    alpha = 1.0 - conf
    n_sim = max(int(n_simulations), 100)
    rng = np.random.default_rng(int(seed))
    if r.size == 0:
        mu, sigma, value = 0.0, 0.0, 0.0
    else:
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        h = max(int(horizon), 1)
        shocks = rng.normal(mu, max(sigma, 1e-12), size=(n_sim, h)).sum(axis=1)
        q = float(np.quantile(shocks, alpha))
        tail = shocks[shocks <= q]
        es = float(max(-np.mean(tail), 0.0)) if tail.size else float(max(-q, 0.0))
        value = es
    return RiskMeasure(
        name="cvar",
        value=value,
        unit="return",
        confidence=conf,
        horizon=int(horizon),
        method="monte_carlo",
        parameters={
            "mu": mu,
            "sigma": sigma,
            "n_obs": int(r.size),
            "n_simulations": n_sim,
            "seed": int(seed),
            "alpha": alpha,
        },
    )
