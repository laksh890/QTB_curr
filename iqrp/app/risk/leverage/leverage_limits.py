"""Leverage clipping / hard caps."""

from __future__ import annotations

import numpy as np

from iqrp.app.risk.base import RiskMeasure


def clip_leverage(
    leverage: float,
    *,
    min_leverage: float = 0.0,
    max_leverage: float = 2.0,
) -> RiskMeasure:
    """Hard-clip leverage into [min, max]. Confidence cannot bypass this."""
    lo = float(min_leverage)
    hi = max(float(max_leverage), lo)
    raw = float(leverage) if np.isfinite(leverage) else 0.0
    clipped = float(np.clip(raw, lo, hi))
    return RiskMeasure(
        name="clip_leverage",
        value=clipped,
        unit="leverage",
        method="hard_clip",
        parameters={
            "raw": raw,
            "min_leverage": lo,
            "max_leverage": hi,
            "was_clipped": clipped != raw,
        },
    )
