"""Flat risk aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, RiskState


def _extract_value(item: Any) -> float:
    if isinstance(item, RiskMeasure):
        return float(item.value)
    if isinstance(item, dict):
        if "value" in item:
            return float(item["value"])
        if "loss" in item:
            return float(item["loss"])
        if "score" in item:
            return float(item["score"])
    return float(item)


def aggregate_risks(
    measures: dict[str, Any] | list[Any],
    *,
    weights: dict[str, float] | list[float] | None = None,
    method: str = "weighted_sum",
) -> dict[str, Any]:
    """Aggregate heterogeneous risk measures into a single score.

    ``method``: weighted_sum | max | rms
    """
    if isinstance(measures, dict):
        keys = list(measures.keys())
        values = np.array([_extract_value(measures[k]) for k in keys], dtype=np.float64)
        if weights is None:
            w = np.full(len(keys), 1.0 / max(len(keys), 1))
        elif isinstance(weights, dict):
            w = np.array([float(weights.get(k, 0.0)) for k in keys], dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size != len(keys):
                tmp = np.zeros(len(keys))
                m = min(w.size, len(keys))
                tmp[:m] = w[:m]
                w = tmp
    else:
        keys = [f"m{i}" for i in range(len(measures))]
        values = np.array([_extract_value(m) for m in measures], dtype=np.float64)
        if weights is None:
            w = np.full(len(keys), 1.0 / max(len(keys), 1))
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size != len(keys):
                tmp = np.zeros(len(keys))
                m = min(w.size, len(keys))
                tmp[:m] = w[:m]
                w = tmp

    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    w = w / s if s > 0 else np.full(w.size, 1.0 / max(w.size, 1))

    m = str(method).lower()
    if m == "max":
        agg = float(np.max(values)) if values.size else 0.0
    elif m == "rms":
        agg = float(np.sqrt(np.sum(w * values ** 2))) if values.size else 0.0
    else:
        agg = float(np.dot(w, values)) if values.size else 0.0
        m = "weighted_sum"

    # Map aggregate magnitude to a coarse state (caller can override)
    if agg >= 0.20:
        state = RiskState.TRADING_HALT
    elif agg >= 0.15:
        state = RiskState.CAPITAL_PRESERVATION
    elif agg >= 0.10:
        state = RiskState.REDUCED_RISK
    elif agg >= 0.05:
        state = RiskState.CAUTION
    else:
        state = RiskState.NORMAL

    return {
        "name": "aggregate_risks",
        "value": agg,
        "method": m,
        "risk_state": state.value,
        "components": {keys[i]: float(values[i]) for i in range(len(keys))},
        "weights": {keys[i]: float(w[i]) for i in range(len(keys))},
        "measure": RiskMeasure(
            name="aggregate_risk",
            value=agg,
            unit="score",
            method=m,
            parameters={"n_components": len(keys)},
        ).to_dict(),
    }
