"""Concentration risk metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_weights


def herfindahl(weights: Any) -> RiskMeasure:
    """Herfindahl-Hirschman index on absolute normalized weights."""
    w = as_weights(weights)
    if w.size == 0:
        return RiskMeasure(name="herfindahl", value=0.0, unit="index", method="hhi")
    abs_w = np.abs(w)
    total = float(np.sum(abs_w))
    if total <= 0:
        shares = np.zeros_like(abs_w)
    else:
        shares = abs_w / total
    value = float(np.sum(shares**2))
    return RiskMeasure(
        name="herfindahl",
        value=value,
        unit="index",
        method="hhi",
        parameters={
            "n_assets": int(w.size),
            "effective_n": float(1.0 / value) if value > 0 else 0.0,
        },
    )


def max_weight(weights: Any) -> RiskMeasure:
    w = as_weights(weights)
    value = float(np.max(np.abs(w))) if w.size else 0.0
    idx = int(np.argmax(np.abs(w))) if w.size else -1
    return RiskMeasure(
        name="max_weight",
        value=value,
        unit="fraction",
        method="abs_max",
        parameters={"index": idx, "n_assets": int(w.size)},
    )


def concentration_risk(weights: Any) -> dict[str, Any]:
    """Composite concentration diagnostics."""
    hhi = herfindahl(weights)
    mw = max_weight(weights)
    # Score: 0 = diversified, 1 = fully concentrated
    n = max(int(as_weights(weights).size), 1)
    min_hhi = 1.0 / n
    score = float(np.clip((hhi.value - min_hhi) / max(1.0 - min_hhi, 1e-12), 0.0, 1.0))
    return {
        "name": "concentration_risk",
        "score": score,
        "herfindahl": hhi.to_dict(),
        "max_weight": mw.to_dict(),
    }
