"""Portfolio exposure metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_weights


def gross_exposure(weights: Any) -> RiskMeasure:
    w = as_weights(weights)
    value = float(np.sum(np.abs(w))) if w.size else 0.0
    return RiskMeasure(name="gross_exposure", value=value, unit="fraction", method="l1")


def net_exposure(weights: Any) -> RiskMeasure:
    w = as_weights(weights)
    value = float(np.sum(w)) if w.size else 0.0
    return RiskMeasure(name="net_exposure", value=value, unit="fraction", method="sum")


def long_exposure(weights: Any) -> RiskMeasure:
    w = as_weights(weights)
    value = float(np.sum(w[w > 0])) if w.size else 0.0
    return RiskMeasure(name="long_exposure", value=value, unit="fraction", method="positive_sum")


def short_exposure(weights: Any) -> RiskMeasure:
    w = as_weights(weights)
    value = float(np.sum(np.abs(w[w < 0]))) if w.size else 0.0
    return RiskMeasure(
        name="short_exposure", value=value, unit="fraction", method="abs_negative_sum"
    )


def exposure_summary(weights: Any) -> dict[str, Any]:
    """Aggregate exposure report."""
    return {
        "name": "exposure_summary",
        "gross": gross_exposure(weights).to_dict(),
        "net": net_exposure(weights).to_dict(),
        "long": long_exposure(weights).to_dict(),
        "short": short_exposure(weights).to_dict(),
    }
