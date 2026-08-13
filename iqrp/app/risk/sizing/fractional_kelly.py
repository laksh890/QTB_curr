"""Fractional Kelly sizing."""

from __future__ import annotations

import numpy as np

from iqrp.app.risk.base import RiskMeasure
from iqrp.app.risk.sizing.kelly import kelly_fraction


def fractional_kelly(
    *,
    edge: float,
    odds: float = 1.0,
    win_prob: float | None = None,
    variance: float | None = None,
    fraction: float = 0.25,
    max_kelly: float = 0.5,
) -> RiskMeasure:
    """Fractional Kelly = fraction * capped_kelly; still respects max_kelly."""
    frac = float(np.clip(fraction, 0.0, 1.0))
    base = kelly_fraction(
        edge=edge,
        odds=odds,
        win_prob=win_prob,
        variance=variance,
        max_kelly=max_kelly,
    )
    value = float(np.clip(frac * base.value, 0.0, max(float(max_kelly), 0.0)))
    return RiskMeasure(
        name="fractional_kelly",
        value=value,
        unit="fraction",
        method="fractional_kelly",
        parameters={
            "fraction": frac,
            "max_kelly": float(max_kelly),
            "full_capped_kelly": base.value,
            "raw_kelly": base.parameters.get("raw_kelly"),
            "edge": float(edge),
            "odds": float(odds),
            "win_prob": win_prob,
            "variance": variance,
        },
    )
