"""Portfolio visualization payloads (no matplotlib dependency)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def weights_payload(
    weights: Any,
    *,
    names: Sequence[str] | None = None,
    title: str = "weights",
) -> dict[str, Any]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    labels = list(names) if names is not None else [f"asset_{i}" for i in range(w.size)]
    if len(labels) < w.size:
        labels = labels + [f"asset_{i}" for i in range(len(labels), w.size)]
    return {
        "name": "weights_chart",
        "title": title,
        "type": "bar",
        "labels": labels[: w.size],
        "values": w.tolist(),
        "gross": float(np.sum(np.abs(w))) if w.size else 0.0,
        "net": float(np.sum(w)) if w.size else 0.0,
    }


def risk_contribution_payload(
    contributions: Any,
    *,
    names: Sequence[str] | None = None,
    title: str = "risk_contribution",
    percent: bool = False,
) -> dict[str, Any]:
    if isinstance(contributions, dict):
        vals = contributions.get("percent" if percent else "values", contributions.get("values", []))
        if percent and "percent" in contributions and contributions["percent"] and max(
            abs(float(x)) for x in contributions["percent"]
        ) <= 1.0 + 1e-9:
            # values already fractions — scale for display if requested as percent points
            c = np.asarray(contributions["percent"], dtype=np.float64)
            values = (c * 100.0).tolist() if percent else c.tolist()
        else:
            values = list(vals)
    else:
        values = np.asarray(contributions, dtype=np.float64).reshape(-1).tolist()
    n = len(values)
    labels = list(names) if names is not None else [f"asset_{i}" for i in range(n)]
    return {
        "name": "risk_contribution_chart",
        "title": title,
        "type": "bar",
        "labels": labels[:n],
        "values": values,
        "unit": "percent" if percent else "volatility",
    }


def turnover_payload(
    weights_old: Any,
    weights_new: Any,
    *,
    names: Sequence[str] | None = None,
    title: str = "turnover",
) -> dict[str, Any]:
    a = np.asarray(weights_old, dtype=np.float64).reshape(-1)
    b = np.asarray(weights_new, dtype=np.float64).reshape(-1)
    n = max(a.size, b.size)
    aa = np.zeros(n)
    bb = np.zeros(n)
    aa[: a.size] = a
    bb[: b.size] = b
    delta = bb - aa
    turnover = 0.5 * float(np.sum(np.abs(delta)))
    labels = list(names) if names is not None else [f"asset_{i}" for i in range(n)]
    return {
        "name": "turnover_chart",
        "title": title,
        "type": "bar",
        "labels": labels[:n],
        "values": delta.tolist(),
        "abs_values": np.abs(delta).tolist(),
        "turnover": turnover,
        "weights_old": aa.tolist(),
        "weights_new": bb.tolist(),
    }


def portfolio_viz_bundle(
    weights: Any,
    *,
    weights_old: Any | None = None,
    risk_contribution: Any | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bundle of chart-ready payloads for dashboards."""
    out: dict[str, Any] = {
        "name": "portfolio_viz_bundle",
        "weights": weights_payload(weights, names=names),
    }
    if risk_contribution is not None:
        out["risk_contribution"] = risk_contribution_payload(
            risk_contribution, names=names
        )
    if weights_old is not None:
        out["turnover"] = turnover_payload(weights_old, weights, names=names)
    return out
