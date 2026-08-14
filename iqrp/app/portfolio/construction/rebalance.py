"""Rebalance triggers, bands, and trade generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

TriggerKind = Literal[
    "scheduled",
    "threshold",
    "risk",
    "drift",
    "regime",
    "drawdown",
    "liquidity",
    "manual",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _as_w(x: Any, n: int | None = None) -> np.ndarray:
    v = np.asarray(x if x is not None else [], dtype=np.float64).reshape(-1)
    if n is not None:
        out = np.zeros(n, dtype=np.float64)
        m = min(n, v.size)
        out[:m] = v[:m]
        return out
    return v


@dataclass(slots=True)
class RebalanceBands:
    """No-trade region around target weights."""

    absolute: float = 0.0
    relative: float = 0.0
    min_trade: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "absolute": float(self.absolute),
            "relative": float(self.relative),
            "min_trade": float(self.min_trade),
        }


@dataclass(slots=True)
class RebalanceTrigger:
    kind: TriggerKind | str
    fired: bool
    reason: str = ""
    value: float | None = None
    threshold: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "fired": bool(self.fired),
            "reason": self.reason,
            "value": None if self.value is None else float(self.value),
            "threshold": None if self.threshold is None else float(self.threshold),
            "meta": dict(self.meta),
        }


@dataclass(slots=True)
class RebalancePlan:
    """Rebalance decision with target trades (target - current)."""

    should_rebalance: bool
    triggers: list[RebalanceTrigger] = field(default_factory=list)
    current_weights: list[float] = field(default_factory=list)
    target_weights: list[float] = field(default_factory=list)
    trades: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    bands: RebalanceBands = field(default_factory=RebalanceBands)
    turnover: float = 0.0
    timestamp: str = field(default_factory=_utc_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_rebalance": bool(self.should_rebalance),
            "triggers": [t.to_dict() for t in self.triggers],
            "current_weights": [float(w) for w in self.current_weights],
            "target_weights": [float(w) for w in self.target_weights],
            "trades": [float(t) for t in self.trades],
            "names": list(self.names),
            "bands": self.bands.to_dict(),
            "turnover": float(self.turnover),
            "timestamp": self.timestamp,
            "meta": dict(self.meta),
        }


def apply_rebalance_bands(
    current: Any,
    target: Any,
    *,
    absolute: float = 0.0,
    relative: float = 0.0,
    min_trade: float = 0.0,
) -> np.ndarray:
    """Return trades after absolute/relative no-trade bands and min-trade filter.

    Within the band, trade is zero (keep current). Outside, trade = target - current.
    """
    cur = _as_w(current)
    tgt = _as_w(target)
    n = max(cur.size, tgt.size)
    cur = _as_w(cur, n)
    tgt = _as_w(tgt, n)
    trades = tgt - cur
    abs_band = max(float(absolute), 0.0)
    rel_band = max(float(relative), 0.0)
    min_t = max(float(min_trade), 0.0)

    for i in range(n):
        band = abs_band + rel_band * abs(float(tgt[i]))
        if abs(float(trades[i])) <= band + 1e-15 or abs(float(trades[i])) < min_t:
            trades[i] = 0.0
    return trades


def evaluate_triggers(
    *,
    current_weights: Any | None = None,
    target_weights: Any | None = None,
    scheduled: bool = False,
    turnover_threshold: float | None = None,
    risk_breach: bool = False,
    risk_metric: float | None = None,
    risk_limit: float | None = None,
    drift: float | None = None,
    drift_threshold: float | None = None,
    regime_change: bool = False,
    drawdown: float | None = None,
    drawdown_threshold: float | None = None,
    liquidity_stress: bool = False,
    liquidity_score: float | None = None,
    liquidity_threshold: float | None = None,
    force: bool = False,
) -> list[RebalanceTrigger]:
    """Evaluate configured rebalance triggers (OR logic at plan level)."""
    triggers: list[RebalanceTrigger] = []

    triggers.append(
        RebalanceTrigger(
            kind="scheduled",
            fired=bool(scheduled),
            reason="scheduled rebalance due" if scheduled else "not on schedule",
        )
    )

    cur = _as_w(current_weights) if current_weights is not None else None
    tgt = _as_w(target_weights) if target_weights is not None else None
    to = None
    if cur is not None and tgt is not None:
        n = max(cur.size, tgt.size)
        to = 0.5 * float(np.sum(np.abs(_as_w(tgt, n) - _as_w(cur, n))))
    thr = float(turnover_threshold) if turnover_threshold is not None else None
    fired_thr = thr is not None and to is not None and to > thr + 1e-12
    triggers.append(
        RebalanceTrigger(
            kind="threshold",
            fired=bool(fired_thr),
            reason="turnover exceeds threshold" if fired_thr else "turnover within threshold",
            value=to,
            threshold=thr,
        )
    )

    risk_fired = bool(risk_breach)
    if risk_limit is not None and risk_metric is not None:
        risk_fired = risk_fired or float(risk_metric) > float(risk_limit) + 1e-12
    triggers.append(
        RebalanceTrigger(
            kind="risk",
            fired=risk_fired,
            reason="risk limit breach" if risk_fired else "risk within limits",
            value=None if risk_metric is None else float(risk_metric),
            threshold=None if risk_limit is None else float(risk_limit),
        )
    )

    d_val = drift
    if d_val is None and cur is not None and tgt is not None:
        n = max(cur.size, tgt.size)
        d_val = float(np.max(np.abs(_as_w(tgt, n) - _as_w(cur, n))))
    d_thr = float(drift_threshold) if drift_threshold is not None else None
    drift_fired = d_thr is not None and d_val is not None and float(d_val) > d_thr + 1e-12
    triggers.append(
        RebalanceTrigger(
            kind="drift",
            fired=bool(drift_fired),
            reason="weight drift exceeds band" if drift_fired else "drift within band",
            value=None if d_val is None else float(d_val),
            threshold=d_thr,
        )
    )

    triggers.append(
        RebalanceTrigger(
            kind="regime",
            fired=bool(regime_change),
            reason="regime change detected" if regime_change else "no regime change",
        )
    )

    dd = None if drawdown is None else float(drawdown)
    dd_thr = None if drawdown_threshold is None else float(drawdown_threshold)
    dd_fired = dd is not None and dd_thr is not None and abs(dd) >= abs(dd_thr) - 1e-12
    triggers.append(
        RebalanceTrigger(
            kind="drawdown",
            fired=bool(dd_fired),
            reason="drawdown trigger" if dd_fired else "drawdown within tolerance",
            value=dd,
            threshold=dd_thr,
        )
    )

    liq_fired = bool(liquidity_stress)
    if liquidity_score is not None and liquidity_threshold is not None:
        liq_fired = liq_fired or float(liquidity_score) < float(liquidity_threshold)
    triggers.append(
        RebalanceTrigger(
            kind="liquidity",
            fired=liq_fired,
            reason="liquidity stress" if liq_fired else "liquidity ok",
            value=None if liquidity_score is None else float(liquidity_score),
            threshold=None if liquidity_threshold is None else float(liquidity_threshold),
        )
    )

    if force:
        triggers.append(RebalanceTrigger(kind="manual", fired=True, reason="forced rebalance"))

    return triggers


def plan_rebalance(
    current_weights: Any,
    target_weights: Any,
    *,
    names: Sequence[str] | None = None,
    bands: RebalanceBands | dict[str, float] | None = None,
    absolute_band: float = 0.0,
    relative_band: float = 0.0,
    min_trade: float = 0.0,
    scheduled: bool = False,
    turnover_threshold: float | None = None,
    risk_breach: bool = False,
    risk_metric: float | None = None,
    risk_limit: float | None = None,
    drift_threshold: float | None = None,
    regime_change: bool = False,
    drawdown: float | None = None,
    drawdown_threshold: float | None = None,
    liquidity_stress: bool = False,
    liquidity_score: float | None = None,
    liquidity_threshold: float | None = None,
    force: bool = False,
    always_if_triggered: bool = True,
) -> RebalancePlan:
    """Build a rebalance plan: evaluate triggers and compute banded trades."""
    if isinstance(bands, dict):
        band_obj = RebalanceBands(
            absolute=float(bands.get("absolute", absolute_band)),
            relative=float(bands.get("relative", relative_band)),
            min_trade=float(bands.get("min_trade", min_trade)),
        )
    elif isinstance(bands, RebalanceBands):
        band_obj = bands
    else:
        band_obj = RebalanceBands(
            absolute=float(absolute_band),
            relative=float(relative_band),
            min_trade=float(min_trade),
        )

    cur = _as_w(current_weights)
    tgt = _as_w(target_weights)
    n = max(cur.size, tgt.size)
    cur = _as_w(cur, n)
    tgt = _as_w(tgt, n)
    name_list = list(names) if names is not None else [f"a{i}" for i in range(n)]
    if len(name_list) != n:
        name_list = [f"a{i}" for i in range(n)]

    # If drift_threshold not set, use absolute band as drift trigger
    d_thr = (
        drift_threshold
        if drift_threshold is not None
        else (band_obj.absolute if band_obj.absolute > 0 else None)
    )

    triggers = evaluate_triggers(
        current_weights=cur,
        target_weights=tgt,
        scheduled=scheduled,
        turnover_threshold=turnover_threshold,
        risk_breach=risk_breach,
        risk_metric=risk_metric,
        risk_limit=risk_limit,
        drift_threshold=d_thr,
        regime_change=regime_change,
        drawdown=drawdown,
        drawdown_threshold=drawdown_threshold,
        liquidity_stress=liquidity_stress,
        liquidity_score=liquidity_score,
        liquidity_threshold=liquidity_threshold,
        force=force,
    )
    any_fired = any(t.fired for t in triggers)
    should = bool(force or (always_if_triggered and any_fired) or scheduled)

    if should:
        trades = apply_rebalance_bands(
            cur,
            tgt,
            absolute=band_obj.absolute,
            relative=band_obj.relative,
            min_trade=band_obj.min_trade,
        )
        # If all trades zeroed by bands but a hard trigger fired, allow full trade
        if float(np.sum(np.abs(trades))) < 1e-15 and any(
            t.fired and t.kind in ("risk", "drawdown", "regime", "manual", "scheduled")
            for t in triggers
        ):
            trades = tgt - cur
            if band_obj.min_trade > 0:
                trades = np.where(np.abs(trades) < band_obj.min_trade, 0.0, trades)
    else:
        trades = np.zeros(n, dtype=np.float64)

    turnover = 0.5 * float(np.sum(np.abs(trades)))
    return RebalancePlan(
        should_rebalance=should and turnover > 1e-15,
        triggers=triggers,
        current_weights=[float(x) for x in cur.tolist()],
        target_weights=[float(x) for x in tgt.tolist()],
        trades=[float(x) for x in trades.tolist()],
        names=name_list,
        bands=band_obj,
        turnover=turnover,
        meta={"any_trigger_fired": any_fired},
    )
