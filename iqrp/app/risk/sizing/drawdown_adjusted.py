"""Drawdown-adjusted position sizing."""

from __future__ import annotations

import numpy as np

from iqrp.app.risk.base import RiskMeasure


def drawdown_adjusted_size(
    *,
    base_size: float,
    current_drawdown: float,
    max_drawdown_limit: float = 0.20,
    floor: float = 0.0,
) -> RiskMeasure:
    """Linearly reduce size as drawdown approaches the hard limit.

    At drawdown >= limit, size collapses to ``floor`` (default 0).
    Hard drawdown limits are never relaxed by confidence.
    """
    dd = max(float(current_drawdown), 0.0)
    limit = max(float(max_drawdown_limit), 1e-12)
    fl = max(float(floor), 0.0)
    scale = float(np.clip(1.0 - dd / limit, 0.0, 1.0))
    size = fl + (float(base_size) - fl) * scale
    if dd >= limit:
        size = fl
    return RiskMeasure(
        name="drawdown_adjusted_size",
        value=float(size),
        unit="size",
        method="drawdown_linear_scale",
        parameters={
            "base_size": float(base_size),
            "current_drawdown": dd,
            "max_drawdown_limit": limit,
            "floor": fl,
            "scale": scale,
        },
    )
