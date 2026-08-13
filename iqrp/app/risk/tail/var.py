"""Value-at-Risk estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from iqrp.app.risk.base import RiskMeasure, as_returns

_ALLOWED_CONF = (0.90, 0.95, 0.99)


def _normalize_confidence(confidence: float) -> float:
    c = float(confidence)
    if c in _ALLOWED_CONF:
        return c
    # Snap to nearest supported level
    return float(min(_ALLOWED_CONF, key=lambda x: abs(x - c)))


def _scale_horizon(var_1d: float, horizon: int) -> float:
    h = max(int(horizon), 1)
    return float(var_1d * np.sqrt(h))


def historical_var(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
) -> RiskMeasure:
    """Historical simulation VaR (positive loss number)."""
    r = as_returns(returns)
    conf = _normalize_confidence(confidence)
    alpha = 1.0 - conf
    if r.size == 0:
        value = 0.0
    else:
        q = float(np.quantile(r, alpha))
        value = _scale_horizon(max(-q, 0.0), horizon)
    return RiskMeasure(
        name="var",
        value=value,
        unit="return",
        confidence=conf,
        horizon=int(horizon),
        method="historical",
        parameters={"n_obs": int(r.size), "alpha": alpha},
    )


def parametric_var(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
) -> RiskMeasure:
    """Gaussian parametric VaR."""
    r = as_returns(returns)
    conf = _normalize_confidence(confidence)
    alpha = 1.0 - conf
    if r.size == 0:
        mu, sigma, value = 0.0, 0.0, 0.0
    else:
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        q = float(stats.norm.ppf(alpha, loc=mu, scale=max(sigma, 1e-12)))
        value = _scale_horizon(max(-q, 0.0), horizon)
    return RiskMeasure(
        name="var",
        value=value,
        unit="return",
        confidence=conf,
        horizon=int(horizon),
        method="parametric",
        parameters={"mu": mu, "sigma": sigma, "n_obs": int(r.size), "alpha": alpha},
    )


def monte_carlo_var(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 5000,
    seed: int = 42,
) -> RiskMeasure:
    """Monte Carlo VaR under i.i.d. normal fitted to historical returns."""
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
        # Simulate horizon aggregate returns as sum of i.i.d. normals
        shocks = rng.normal(mu, max(sigma, 1e-12), size=(n_sim, h)).sum(axis=1)
        q = float(np.quantile(shocks, alpha))
        value = float(max(-q, 0.0))
    return RiskMeasure(
        name="var",
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


def filtered_historical_var(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    lambda_: float = 0.94,
) -> RiskMeasure:
    """Filtered historical simulation (FHS) VaR via EWMA volatility scaling.

    Residuals are standardized by causal EWMA vol, then re-scaled by the
    latest EWMA vol — no future information used.
    """
    r = as_returns(returns)
    conf = _normalize_confidence(confidence)
    alpha = 1.0 - conf
    lam = float(np.clip(lambda_, 1e-6, 1.0 - 1e-6))

    if r.size < 2:
        return RiskMeasure(
            name="var",
            value=0.0,
            unit="return",
            confidence=conf,
            horizon=int(horizon),
            method="filtered_historical",
            parameters={"n_obs": int(r.size), "lambda": lam, "alpha": alpha},
        )

    var = float(r[0] ** 2)
    vols = np.empty(r.size, dtype=np.float64)
    vols[0] = max(np.sqrt(var), 1e-12)
    for t in range(1, r.size):
        var = lam * var + (1.0 - lam) * float(r[t - 1] ** 2)  # causal: use prior return
        vols[t] = max(np.sqrt(var), 1e-12)

    z = r / vols
    z_q = float(np.quantile(z, alpha))
    latest_vol = float(vols[-1])
    value = _scale_horizon(max(-z_q * latest_vol, 0.0), horizon)

    return RiskMeasure(
        name="var",
        value=value,
        unit="return",
        confidence=conf,
        horizon=int(horizon),
        method="filtered_historical",
        parameters={
            "n_obs": int(r.size),
            "lambda": lam,
            "alpha": alpha,
            "latest_vol": latest_vol,
            "z_quantile": z_q,
        },
    )
