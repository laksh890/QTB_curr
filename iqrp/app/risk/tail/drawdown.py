"""Drawdown analytics and risk-state mapping."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, RiskState, as_returns


def drawdown_series(returns: Any) -> np.ndarray:
    """Compute running drawdown series from returns (wealth path based)."""
    r = as_returns(returns)
    if r.size == 0:
        return np.zeros(0, dtype=np.float64)
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - wealth / np.maximum(peak, 1e-12)
    return np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)


def max_drawdown(returns: Any) -> RiskMeasure:
    """Maximum drawdown (positive fraction)."""
    dd = drawdown_series(returns)
    value = float(np.max(dd)) if dd.size else 0.0
    return RiskMeasure(
        name="max_drawdown",
        value=value,
        unit="fraction",
        method="peak_to_trough",
        parameters={"n_obs": int(dd.size)},
    )


def expected_drawdown(returns: Any) -> RiskMeasure:
    """Mean of the drawdown series (average underwater fraction)."""
    dd = drawdown_series(returns)
    value = float(np.mean(dd)) if dd.size else 0.0
    return RiskMeasure(
        name="expected_drawdown",
        value=value,
        unit="fraction",
        method="mean_drawdown",
        parameters={"n_obs": int(dd.size)},
    )


def ulcer_index(returns: Any) -> RiskMeasure:
    """Ulcer index: RMS of drawdowns."""
    dd = drawdown_series(returns)
    value = float(np.sqrt(np.mean(dd**2))) if dd.size else 0.0
    return RiskMeasure(
        name="ulcer_index",
        value=value,
        unit="fraction",
        method="rms_drawdown",
        parameters={"n_obs": int(dd.size)},
    )


def downside_deviation(
    returns: Any,
    *,
    mar: float = 0.0,
) -> RiskMeasure:
    """Downside deviation relative to a minimum acceptable return (MAR)."""
    r = as_returns(returns)
    if r.size == 0:
        value = 0.0
    else:
        downside = np.minimum(r - float(mar), 0.0)
        value = float(np.sqrt(np.mean(downside**2)))
    return RiskMeasure(
        name="downside_deviation",
        value=value,
        unit="volatility",
        method="semideviation",
        parameters={"mar": float(mar), "n_obs": int(r.size)},
    )


def _drawdown_path_stats(returns: Any) -> dict[str, Any]:
    """Peak equity, current duration, and last recovery time from wealth path."""
    r = as_returns(returns)
    if r.size == 0:
        return {
            "peak_equity": 1.0,
            "drawdown_duration": 0,
            "recovery_time": None,
            "wealth": 1.0,
        }
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = 1.0 - wealth / np.maximum(peak, 1e-12)
    underwater = dd > 1e-12
    # Current drawdown duration (bars since last peak)
    duration = 0
    for flag in underwater[::-1]:
        if flag:
            duration += 1
        else:
            break
    # Recovery time of the most recently completed drawdown episode
    recovery_time: int | None = None
    episode_start: int | None = None
    last_completed: int | None = None
    for i, flag in enumerate(underwater):
        if flag and episode_start is None:
            episode_start = i
        elif not flag and episode_start is not None:
            last_completed = i - episode_start
            episode_start = None
    recovery_time = last_completed
    return {
        "peak_equity": float(peak[-1]),
        "drawdown_duration": int(duration),
        "recovery_time": recovery_time,
        "wealth": float(wealth[-1]),
    }


def drawdown_state(
    returns: Any,
    *,
    caution: float = 0.05,
    reduced_risk: float = 0.10,
    capital_preservation: float = 0.15,
    trading_halt: float = 0.20,
) -> dict[str, Any]:
    """Map *current* drawdown onto RiskState; report max drawdown separately.

    Live risk-state transitions are driven by current underwater fraction so a
    recovered book can leave TRADING_HALT / CAPITAL_PRESERVATION. Historical
    maximum drawdown remains in the report for audit and limit checks.
    """
    dd = drawdown_series(returns)
    current = float(dd[-1]) if dd.size else 0.0
    mdd = float(np.max(dd)) if dd.size else 0.0
    path = _drawdown_path_stats(returns)
    level = current

    if level >= trading_halt:
        state = RiskState.TRADING_HALT
    elif level >= capital_preservation:
        state = RiskState.CAPITAL_PRESERVATION
    elif level >= reduced_risk:
        state = RiskState.REDUCED_RISK
    elif level >= caution:
        state = RiskState.CAUTION
    else:
        state = RiskState.NORMAL

    return {
        "name": "drawdown_state",
        "risk_state": state.value,
        "current_drawdown": current,
        "max_drawdown": mdd,
        "peak_equity": path["peak_equity"],
        "drawdown_duration": path["drawdown_duration"],
        "recovery_time": path["recovery_time"],
        "wealth": path["wealth"],
        "thresholds": {
            "caution": float(caution),
            "reduced_risk": float(reduced_risk),
            "capital_preservation": float(capital_preservation),
            "trading_halt": float(trading_halt),
        },
        "measures": {
            "max_drawdown": max_drawdown(returns).to_dict(),
            "expected_drawdown": expected_drawdown(returns).to_dict(),
            "ulcer_index": ulcer_index(returns).to_dict(),
            "downside_deviation": downside_deviation(returns).to_dict(),
        },
    }
