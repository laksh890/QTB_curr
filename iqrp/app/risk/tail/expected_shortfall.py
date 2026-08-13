"""Expected Shortfall aliases and conditional tail expectation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns
from iqrp.app.risk.tail.cvar import historical_cvar, monte_carlo_cvar, parametric_cvar
from iqrp.app.risk.tail.var import _normalize_confidence, _scale_horizon


def expected_shortfall(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    method: str = "historical",
    n_simulations: int = 5000,
    seed: int = 42,
) -> RiskMeasure:
    """Expected Shortfall (alias to CVaR methods)."""
    m = str(method).lower()
    if m in ("parametric", "gaussian", "normal"):
        rm = parametric_cvar(returns, confidence=confidence, horizon=horizon)
    elif m in ("monte_carlo", "mc", "simulation"):
        rm = monte_carlo_cvar(
            returns,
            confidence=confidence,
            horizon=horizon,
            n_simulations=n_simulations,
            seed=seed,
        )
    else:
        rm = historical_cvar(returns, confidence=confidence, horizon=horizon)
    return RiskMeasure(
        name="expected_shortfall",
        value=rm.value,
        unit=rm.unit,
        confidence=rm.confidence,
        horizon=rm.horizon,
        method=rm.method,
        parameters=dict(rm.parameters),
        metadata=dict(rm.metadata),
    )


def conditional_tail_expectation(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    threshold: float | None = None,
) -> RiskMeasure:
    """Conditional tail expectation below a quantile or absolute threshold.

    If ``threshold`` is provided, conditions on returns <= threshold;
    otherwise uses the (1 - confidence) empirical quantile.
    """
    r = as_returns(returns)
    conf = _normalize_confidence(confidence)
    alpha = 1.0 - conf
    if r.size == 0:
        return RiskMeasure(
            name="conditional_tail_expectation",
            value=0.0,
            unit="return",
            confidence=conf,
            horizon=int(horizon),
            method="cte",
            parameters={"n_obs": 0, "alpha": alpha},
        )

    if threshold is None:
        thr = float(np.quantile(r, alpha))
    else:
        thr = float(threshold)

    tail = r[r <= thr]
    if tail.size == 0:
        cte = float(max(-thr, 0.0))
    else:
        cte = float(max(-np.mean(tail), 0.0))
    value = _scale_horizon(cte, horizon)

    return RiskMeasure(
        name="conditional_tail_expectation",
        value=value,
        unit="return",
        confidence=conf,
        horizon=int(horizon),
        method="cte",
        parameters={
            "n_obs": int(r.size),
            "n_tail": int(tail.size),
            "threshold": thr,
            "alpha": alpha,
        },
    )
