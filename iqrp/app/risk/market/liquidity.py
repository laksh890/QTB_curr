"""Liquidity risk metrics: ADV, spread, participation, liquidation time, slippage."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure


def liquidity_risk(
    *,
    position_size: float,
    adv: float,
    spread: float,
    price: float = 1.0,
    volatility: float = 0.0,
    max_participation: float = 0.10,
    impact_coeff: float = 0.1,
    trading_days_year: float = 252.0,
) -> dict[str, Any]:
    """Estimate liquidity risk for a single position.

    Parameters
    ----------
    position_size :
        Absolute notional or shares (same units as ADV * price if notional).
    adv :
        Average daily volume in the same unit basis as ``position_size / price``
        when ``price`` is provided (share ADV), or notional ADV if price=1.
    spread :
        Bid-ask spread as a fraction of mid (e.g. 0.001 = 10 bps).
    price :
        Mid price used to convert notional to shares when needed.
    volatility :
        Daily return volatility used for square-root market impact.
    max_participation :
        Cap on daily ADV participation for liquidation schedule.
    impact_coeff :
        Temporary impact coefficient for square-root model.
    """
    pos = abs(float(position_size))
    adv_v = max(float(adv), 1e-12)
    px = max(float(price), 1e-12)
    spr = max(float(spread), 0.0)
    vol = max(float(volatility), 0.0)
    part_cap = float(np.clip(max_participation, 1e-6, 1.0))
    k = max(float(impact_coeff), 0.0)

    # Treat position_size as notional; shares = notional / price
    shares = pos / px
    participation = shares / adv_v
    daily_capacity = part_cap * adv_v
    time_to_liquidate = float(shares / max(daily_capacity, 1e-12))

    # Square-root temporary impact (fraction of price) + half-spread
    temp_impact = k * vol * float(np.sqrt(max(participation, 0.0)))
    slippage = 0.5 * spr + temp_impact
    slippage_cost = slippage * pos

    adv_coverage = adv_v * px / max(pos, 1e-12)  # days of ADV covering position notional

    measures = {
        "adv": RiskMeasure(
            name="adv",
            value=float(adv_v),
            unit="volume",
            method="input",
        ).to_dict(),
        "spread": RiskMeasure(
            name="spread",
            value=float(spr),
            unit="fraction",
            method="input",
        ).to_dict(),
        "participation": RiskMeasure(
            name="participation",
            value=float(participation),
            unit="fraction",
            method="position/adv",
            parameters={"position_shares": shares, "adv": adv_v},
        ).to_dict(),
        "time_to_liquidate": RiskMeasure(
            name="time_to_liquidate",
            value=time_to_liquidate,
            unit="days",
            method="participation_cap",
            parameters={"max_participation": part_cap},
        ).to_dict(),
        "slippage": RiskMeasure(
            name="slippage_estimate",
            value=float(slippage),
            unit="fraction",
            method="sqrt_impact",
            parameters={"impact_coeff": k, "volatility": vol, "half_spread": 0.5 * spr},
            metadata={"slippage_cost": float(slippage_cost)},
        ).to_dict(),
        "adv_coverage": RiskMeasure(
            name="adv_coverage",
            value=float(adv_coverage),
            unit="ratio",
            method="adv_notional/position",
        ).to_dict(),
    }

    # Composite score in [0, 1]: higher = more liquid / safer
    # Penalize high participation, long liquidation, wide spread
    score = float(
        np.clip(
            1.0 / (1.0 + participation + 0.1 * time_to_liquidate + 10.0 * spr + 5.0 * temp_impact),
            0.0,
            1.0,
        )
    )

    return {
        "name": "liquidity_risk",
        "score": score,
        "measures": measures,
        "parameters": {
            "position_size": pos,
            "price": px,
            "max_participation": part_cap,
            "impact_coeff": k,
            "trading_days_year": trading_days_year,
        },
    }
