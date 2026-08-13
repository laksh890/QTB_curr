"""Strategy capacity from ADV participation and turnover.

``max_capital ≈ ADV * max_participation / turnover``

where ``turnover`` is the fraction of AUM traded per period (e.g. daily).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_capacity(
    *,
    turnover: float,
    adv: float,
    max_participation: float = 0.1,
    periods_per_year: float = 252.0,
    annualize_turnover: bool = False,
) -> dict[str, float]:
    """Estimate max deployable capital under an ADV participation cap.

    Parameters
    ----------
    turnover:
        Portfolio turnover as a fraction of AUM **per period** (same period as
        ADV), unless ``annualize_turnover=True`` in which case it is converted
        to per-period by dividing by ``periods_per_year``.
    adv:
        Average daily (period) volume in currency units.
    max_participation:
        Maximum fraction of ADV the strategy may trade.

    Returns
    -------
    dict with ``max_capital`` and inputs.
    """
    to = float(turnover)
    if annualize_turnover:
        to = to / max(float(periods_per_year), 1e-12)
    to = max(to, 1e-12)
    part = float(np.clip(max_participation, 1e-12, 1.0))
    adv_v = max(float(adv), 0.0)
    max_capital = adv_v * part / to
    return {
        "max_capital": float(max_capital),
        "adv": adv_v,
        "turnover": float(to),
        "max_participation": part,
        "daily_trade_budget": float(adv_v * part),
        "capacity_formula": "adv * max_participation / turnover",
    }


def capacity_decay(
    capital: Any,
    *,
    max_capital: float,
    decay_power: float = 1.0,
) -> np.ndarray:
    """Capacity decay factor in (0, 1]: ``(1 + capital/max_capital)^(-power)``."""
    c = np.maximum(np.asarray(capital, dtype=np.float64), 0.0)
    mc = max(float(max_capital), 1e-12)
    p = max(float(decay_power), 0.0)
    return (1.0 + c / mc) ** (-p)
