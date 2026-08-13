"""Drawdown-aware capital scale factors from ``drawdown_state``."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.tail.drawdown import drawdown_state


# Maps RiskState → capital scale (confidence cannot expand beyond 1.0)
_STATE_SCALES: dict[str, float] = {
    "NORMAL": 1.0,
    "CAUTION": 0.8,
    "REDUCED_RISK": 0.5,
    "CAPITAL_PRESERVATION": 0.25,
    "TRADING_HALT": 0.0,
}


def drawdown_scale_from_state(
    state: dict[str, Any] | str,
    *,
    state_scales: dict[str, float] | None = None,
) -> float:
    """Map a drawdown_state result (or risk_state string) to a scale in [0, 1]."""
    scales = dict(_STATE_SCALES)
    if state_scales:
        scales.update({str(k).upper(): float(v) for k, v in state_scales.items()})
    if isinstance(state, dict):
        key = str(state.get("risk_state", "NORMAL")).upper()
        current = float(state.get("current_drawdown", 0.0) or 0.0)
        thr = state.get("thresholds") or {}
        halt = float(thr.get("trading_halt", 0.20))
        # Continuous interpolant under the discrete state ceiling
        continuous = float(np.clip(1.0 - current / max(halt, 1e-12), 0.0, 1.0))
        discrete = float(np.clip(scales.get(key, 0.5), 0.0, 1.0))
        return float(min(continuous, discrete))
    key = str(state).upper()
    return float(np.clip(scales.get(key, 0.5), 0.0, 1.0))


def drawdown_scales(
    names: list[str],
    *,
    returns: np.ndarray | None = None,
    drawdowns: np.ndarray | list[float] | None = None,
    caution: float = 0.05,
    reduced_risk: float = 0.10,
    capital_preservation: float = 0.15,
    trading_halt: float = 0.20,
    state_scales: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Per-name drawdown scales. Prefer returns → drawdown_state; else use drawdowns levels."""
    n = len(names)
    scales: dict[str, float] = {}
    states: dict[str, Any] = {}

    # Explicit point-in-time drawdown levels take precedence over path-derived state
    if drawdowns is not None:
        dd = np.asarray(drawdowns, dtype=np.float64).ravel()
        for i, name in enumerate(names):
            level = float(dd[i]) if i < dd.size and np.isfinite(dd[i]) else 0.0
            level = max(level, 0.0)
            if level >= trading_halt:
                key = "TRADING_HALT"
            elif level >= capital_preservation:
                key = "CAPITAL_PRESERVATION"
            elif level >= reduced_risk:
                key = "REDUCED_RISK"
            elif level >= caution:
                key = "CAUTION"
            else:
                key = "NORMAL"
            st = {
                "risk_state": key,
                "current_drawdown": level,
                "thresholds": {
                    "caution": caution,
                    "reduced_risk": reduced_risk,
                    "capital_preservation": capital_preservation,
                    "trading_halt": trading_halt,
                },
            }
            states[name] = st
            scales[name] = drawdown_scale_from_state(st, state_scales=state_scales)
    elif returns is not None:
        r = np.asarray(returns, dtype=np.float64)
        if r.ndim == 1:
            r = r.reshape(-1, 1)
        for i, name in enumerate(names):
            if i >= r.shape[1]:
                scales[name] = float(state_scales.get("NORMAL", 1.0) if state_scales else 1.0)
                continue
            st = drawdown_state(
                r[:, i],
                caution=caution,
                reduced_risk=reduced_risk,
                capital_preservation=capital_preservation,
                trading_halt=trading_halt,
            )
            states[name] = st
            scales[name] = drawdown_scale_from_state(st, state_scales=state_scales)
    else:
        # No drawdown info → neutral (do not invent stress)
        for name in names:
            scales[name] = 1.0

    # Ensure length for any missing names
    for name in names:
        scales.setdefault(name, 1.0)

    return {
        "name": "drawdown_scales",
        "scales": scales,
        "states": states,
        "n": n,
    }


def apply_drawdown_scales(
    weights: np.ndarray | list[float],
    scales: dict[str, float],
    *,
    names: list[str],
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    s = np.asarray([float(scales.get(names[i], 1.0)) for i in range(len(names))], dtype=np.float64)
    s = np.clip(s, 0.0, 1.0)
    out = np.maximum(w, 0.0) * s
    tot = float(np.sum(out))
    if tot > 1e-12:
        out = out / tot
    return out
