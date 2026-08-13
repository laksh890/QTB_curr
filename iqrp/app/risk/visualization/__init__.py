"""Risk visualization helpers — structured payloads for dashboards (no UI deps)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskReport, RiskState, as_returns
from iqrp.app.risk.tail.drawdown import drawdown_series


def drawdown_chart(returns: Any) -> dict[str, Any]:
    r = as_returns(returns)
    dd = drawdown_series(r)
    return {
        "name": "drawdown_chart",
        "x": list(range(int(dd.size))),
        "y": dd.tolist(),
        "current": float(dd[-1]) if dd.size else 0.0,
        "max": float(np.max(dd)) if dd.size else 0.0,
    }


def var_histogram(returns: Any, *, bins: int = 40) -> dict[str, Any]:
    r = as_returns(returns)
    if r.size == 0:
        return {"name": "var_histogram", "bins": [], "counts": [], "n_obs": 0}
    counts, edges = np.histogram(r, bins=max(int(bins), 5))
    return {
        "name": "var_histogram",
        "bins": edges.tolist(),
        "counts": counts.tolist(),
        "n_obs": int(r.size),
        "mean": float(np.mean(r)),
        "std": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
    }


def exposure_bars(weights: Any) -> dict[str, Any]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return {
        "name": "exposure_bars",
        "labels": [f"asset_{i}" for i in range(w.size)],
        "values": w.tolist(),
        "gross": float(np.sum(np.abs(w))),
        "net": float(np.sum(w)),
    }


def risk_state_timeline(states: list[str | RiskState]) -> dict[str, Any]:
    vals = [s.value if isinstance(s, RiskState) else str(s) for s in states]
    return {"name": "risk_state_timeline", "states": vals, "n": len(vals)}


def report_panels(report: RiskReport | dict[str, Any]) -> dict[str, Any]:
    if isinstance(report, RiskReport):
        payload = report.to_dict()
    else:
        payload = dict(report)
    return {
        "name": "risk_report_panels",
        "risk_state": payload.get("risk_state"),
        "panels": {
            "portfolio": payload.get("portfolio_risk", {}),
            "tail": payload.get("tail_risk", {}),
            "liquidity": payload.get("liquidity_risk", {}),
            "concentration": payload.get("concentration", {}),
            "drawdown": payload.get("drawdown", {}),
            "stress": payload.get("stress", {}),
            "breaches": payload.get("breaches", []),
        },
    }
