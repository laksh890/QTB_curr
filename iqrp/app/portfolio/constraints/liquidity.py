"""Liquidity constraints: ADV, participation, time-to-liquidate."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)
from iqrp.app.risk.market.liquidity import liquidity_risk


def _as_vec(x: Any, n: int, default: float = 1.0) -> np.ndarray:
    if x is None:
        return np.full(n, float(default), dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    out = np.full(n, float(default), dtype=np.float64)
    m = min(n, arr.size)
    out[:m] = arr[:m]
    return out


def check_liquidity_constraints(
    weights: Any,
    *,
    adv: Any | None = None,
    spreads: Any | None = None,
    prices: Any | None = None,
    vols: Any | None = None,
    capital: float = 1.0,
    max_participation: float | None = None,
    max_ttl: float | None = None,
    min_adv_coverage: float | None = None,
    max_participation_rate: float | None = None,
    impact_coeff: float = 0.1,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    """ADV / participation / TTL checks using ``liquidity_risk`` per name.

    Hard liquidity constraints are never auto-relaxed — only reported.
    """
    w = as_weights(weights)
    n = int(w.size)
    if n == 0 or adv is None:
        return []

    part_cap = max_participation if max_participation is not None else max_participation_rate
    if part_cap is None and max_ttl is None and min_adv_coverage is None:
        return []

    adv_v = _as_vec(adv, n, default=1e12)
    spr = _as_vec(spreads, n, default=0.0)
    px = _as_vec(prices, n, default=1.0)
    vol = _as_vec(vols, n, default=0.0)
    cap = max(float(capital), 0.0)
    out: list[ConstraintViolation] = []
    part_limit = float(part_cap) if part_cap is not None else 0.10

    for i in range(n):
        notional = abs(float(w[i])) * cap
        if notional <= 1e-12:
            continue
        lr = liquidity_risk(
            position_size=notional,
            adv=float(max(adv_v[i], 1e-12)),
            spread=float(max(spr[i], 0.0)),
            price=float(max(px[i], 1e-12)),
            volatility=float(max(vol[i], 0.0)),
            max_participation=part_limit,
            impact_coeff=float(impact_coeff),
        )
        measures = lr.get("measures", {})
        participation = float(measures.get("participation", {}).get("value", 0.0))
        ttl = float(measures.get("time_to_liquidate", {}).get("value", 0.0))
        adv_cov = float(measures.get("adv_coverage", {}).get("value", 0.0))

        if part_cap is not None and participation > float(part_cap) + 1e-12:
            out.append(
                make_violation(
                    "max_participation",
                    observed=participation,
                    threshold=float(part_cap),
                    severity=severity,
                    reason=(
                        f"participation[{i}]={participation:.6g} exceeds "
                        f"max_participation {float(part_cap):.6g}"
                    ),
                    scope="position",
                    metadata={"index": int(i), "liquidity_risk": lr},
                )
            )
        if max_ttl is not None and ttl > float(max_ttl) + 1e-12:
            out.append(
                make_violation(
                    "max_ttl",
                    observed=ttl,
                    threshold=float(max_ttl),
                    severity=severity,
                    reason=f"TTL[{i}]={ttl:.6g} days exceeds max_ttl {float(max_ttl):.6g}",
                    scope="position",
                    metadata={"index": int(i), "liquidity_risk": lr},
                )
            )
        if min_adv_coverage is not None and adv_cov < float(min_adv_coverage) - 1e-12:
            out.append(
                make_violation(
                    "min_adv_coverage",
                    observed=adv_cov,
                    threshold=float(min_adv_coverage),
                    severity=severity,
                    reason=(
                        f"adv_coverage[{i}]={adv_cov:.6g} below "
                        f"min_adv_coverage {float(min_adv_coverage):.6g}"
                    ),
                    scope="position",
                    metadata={"index": int(i), "liquidity_risk": lr},
                )
            )
    return out
