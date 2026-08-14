"""Tail risk metrics for backtests (VaR / CVaR / ES / worst periods).

Prefers ``iqrp.app.risk.tail`` when importable; otherwise uses local
historical estimators so the backtesting package stays self-contained.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.returns import as_returns, daily_returns

__all__ = [
    "conditional_value_at_risk",
    "expected_shortfall",
    "summarize_tail",
    "tail_loss",
    "value_at_risk",
    "worst_day",
    "worst_month",
    "worst_week",
]


def _risk_value(measure: Any) -> float:
    if hasattr(measure, "value"):
        return float(measure.value)
    return float(measure)


def _try_risk_tail():
    """Import risk.tail submodules without requiring optional heavy deps."""
    try:
        from iqrp.app.risk.tail import (
            expected_shortfall as _es,
            filtered_historical_var,
            historical_cvar,
            historical_var,
            monte_carlo_cvar,
            monte_carlo_var,
            parametric_cvar,
            parametric_var,
        )

        return {
            "historical_var": historical_var,
            "parametric_var": parametric_var,
            "monte_carlo_var": monte_carlo_var,
            "filtered_historical_var": filtered_historical_var,
            "historical_cvar": historical_cvar,
            "parametric_cvar": parametric_cvar,
            "monte_carlo_cvar": monte_carlo_cvar,
            "expected_shortfall": _es,
        }
    except Exception:
        return None


def _local_var(returns: Any, *, confidence: float = 0.95, horizon: int = 1) -> float:
    r = as_returns(returns)
    if r.size == 0:
        return 0.0
    alpha = 1.0 - float(confidence)
    q = float(np.quantile(r, alpha))
    return float(max(-q, 0.0) * np.sqrt(max(int(horizon), 1)))


def _local_cvar(returns: Any, *, confidence: float = 0.95, horizon: int = 1) -> float:
    r = as_returns(returns)
    if r.size == 0:
        return 0.0
    alpha = 1.0 - float(confidence)
    q = float(np.quantile(r, alpha))
    tail = r[r <= q]
    es = float(max(-np.mean(tail), 0.0)) if tail.size else float(max(-q, 0.0))
    return float(es * np.sqrt(max(int(horizon), 1)))


def value_at_risk(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    method: str = "historical",
) -> float:
    """VaR as a positive loss number."""
    rt = _try_risk_tail()
    if rt is not None:
        m = str(method).lower()
        if m in ("parametric", "gaussian", "normal"):
            rm = rt["parametric_var"](returns, confidence=confidence, horizon=horizon)
        elif m in ("monte_carlo", "mc", "simulation"):
            rm = rt["monte_carlo_var"](returns, confidence=confidence, horizon=horizon)
        elif m in ("filtered", "fhs", "filtered_historical"):
            rm = rt["filtered_historical_var"](returns, confidence=confidence, horizon=horizon)
        else:
            rm = rt["historical_var"](returns, confidence=confidence, horizon=horizon)
        return _risk_value(rm)
    return _local_var(returns, confidence=confidence, horizon=horizon)


def conditional_value_at_risk(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    method: str = "historical",
) -> float:
    """CVaR / Expected Shortfall as a positive loss number."""
    rt = _try_risk_tail()
    if rt is not None:
        m = str(method).lower()
        if m in ("parametric", "gaussian", "normal"):
            rm = rt["parametric_cvar"](returns, confidence=confidence, horizon=horizon)
        elif m in ("monte_carlo", "mc", "simulation"):
            rm = rt["monte_carlo_cvar"](returns, confidence=confidence, horizon=horizon)
        else:
            rm = rt["historical_cvar"](returns, confidence=confidence, horizon=horizon)
        return _risk_value(rm)
    return _local_cvar(returns, confidence=confidence, horizon=horizon)


def expected_shortfall(
    returns: Any,
    *,
    confidence: float = 0.95,
    horizon: int = 1,
    method: str = "historical",
) -> float:
    """Expected Shortfall (alias of CVaR)."""
    rt = _try_risk_tail()
    if rt is not None:
        return _risk_value(
            rt["expected_shortfall"](returns, confidence=confidence, horizon=horizon, method=method)
        )
    return conditional_value_at_risk(returns, confidence=confidence, horizon=horizon, method=method)


def tail_loss(
    returns: Any,
    *,
    confidence: float = 0.95,
) -> float:
    """Mean loss in the left tail (positive number)."""
    return conditional_value_at_risk(returns, confidence=confidence, method="historical")


def worst_day(returns: Any) -> float:
    """Worst single-period return (typically daily)."""
    r = as_returns(returns)
    return float(np.min(r)) if r.size else 0.0


def worst_week(returns: Any, *, periods_per_week: int = 5) -> float:
    """Worst compounded week within the sample."""
    weeks = daily_returns(returns, bars_per_day=max(int(periods_per_week), 1))
    return float(np.min(weeks)) if weeks.size else 0.0


def worst_month(returns: Any, *, periods_per_month: int = 21) -> float:
    """Worst compounded month within the sample."""
    months = daily_returns(returns, bars_per_day=max(int(periods_per_month), 1))
    return float(np.min(months)) if months.size else 0.0


def summarize_tail(
    returns: Any,
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Tail risk summary."""
    return {
        "var": value_at_risk(returns, confidence=confidence),
        "cvar": conditional_value_at_risk(returns, confidence=confidence),
        "expected_shortfall": expected_shortfall(returns, confidence=confidence),
        "tail_loss": tail_loss(returns, confidence=confidence),
        "worst_day": worst_day(returns),
        "worst_week": worst_week(returns),
        "worst_month": worst_month(returns),
    }
