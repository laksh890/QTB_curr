"""Horizon Research Score and robust-horizon selection.

Scoring formula (documented, weights configurable)
-------------------------------------------------
Let components be normalized to roughly [0, 1] where higher is better:

    score = w_sharpe * norm(net_sharpe)
          + w_expectancy * norm(net_expectancy)
          + w_drawdown * (1 - norm(max_drawdown))
          + w_stability * stability
          + w_oos * oos_score
          + w_turnover * (1 - norm(turnover))   # lower turnover preferred
          + w_costs * (1 - cost_ratio)
          + w_trades * trade_sufficiency
          + w_capacity * capacity_score
          + w_stats * statistical_confidence

Default weights sum to 1.0. Highest return alone is NOT the ranking objective.
BEST ROBUST HORIZON requires configurable OOS / drawdown / cost / trade-count
gates — distinct from BEST IN-SAMPLE HORIZON (raw score without gates).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from iqrp.app.backtesting.horizon.types import HorizonResult, HorizonStatus


DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "net_sharpe": 0.20,
    "net_expectancy": 0.10,
    "drawdown": 0.15,
    "stability": 0.10,
    "oos": 0.20,
    "turnover": 0.05,
    "costs": 0.05,
    "trade_count": 0.05,
    "capacity": 0.05,
    "statistical_confidence": 0.05,
}


DEFAULT_ROBUST_GATES: dict[str, Any] = {
    "min_oos_expectancy": 0.0,
    "min_oos_sharpe": 0.0,
    "max_drawdown": 0.35,
    "max_cost_to_gross": 0.90,
    "min_trades": 5,
    "min_neighborhood_stability": 0.4,
    "require_positive_net_expectancy": True,
}


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _norm_sharpe(s: float) -> float:
    # map roughly [-1, 3] → [0, 1]
    return _clamp01((float(s) + 1.0) / 4.0)


def _norm_dd(dd: float) -> float:
    return _clamp01(abs(float(dd)))


def _norm_turn(t: float) -> float:
    # annualized turnover 0..50 → 0..1
    return _clamp01(abs(float(t)) / 50.0)


@dataclass
class HorizonScoreWeights:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))

    def normalized(self) -> dict[str, float]:
        w = {k: float(v) for k, v in self.weights.items()}
        s = sum(w.values()) or 1.0
        return {k: v / s for k, v in w.items()}


def compute_horizon_research_score(
    metrics: Mapping[str, Any],
    *,
    oos: Mapping[str, Any] | None = None,
    costs: Mapping[str, Any] | None = None,
    turnover: Mapping[str, Any] | None = None,
    capacity: Mapping[str, Any] | None = None,
    neighborhood: Mapping[str, Any] | None = None,
    multiple_testing: Mapping[str, Any] | None = None,
    weights: Mapping[str, float] | HorizonScoreWeights | None = None,
) -> dict[str, Any]:
    """Compute Horizon Research Score (not max-return)."""
    if isinstance(weights, HorizonScoreWeights):
        w = weights.normalized()
    elif weights:
        w = HorizonScoreWeights(dict(weights)).normalized()
    else:
        w = HorizonScoreWeights().normalized()

    oos = dict(oos or {})
    costs = dict(costs or {})
    turnover = dict(turnover or {})
    capacity = dict(capacity or {})
    neighborhood = dict(neighborhood or {})
    mt = dict(multiple_testing or {})

    net_sharpe = float(metrics.get("net_sharpe", metrics.get("sharpe", 0.0)) or 0.0)
    expectancy = float(metrics.get("expectancy_per_trade", 0.0) or 0.0)
    dd = float(metrics.get("maximum_drawdown", 0.0) or 0.0)
    stability = float(
        neighborhood.get("stability_score", metrics.get("stability", 0.5)) or 0.5
    )
    oos_sharpe = float(oos.get("net_sharpe", oos.get("sharpe", 0.0)) or 0.0)
    oos_score = _norm_sharpe(oos_sharpe) if oos else 0.0
    ann_to = float(turnover.get("annualized_turnover", metrics.get("turnover", 0.0)) or 0.0)
    gross = float(metrics.get("total_return_gross", 0.0) or 0.0)
    tcost = float(costs.get("transaction_costs", 0.0) or 0.0)
    cost_ratio = _clamp01(abs(tcost) / abs(gross)) if abs(gross) > 1e-12 else _clamp01(abs(tcost))
    n_trades = int(metrics.get("trade_count", 0) or 0)
    trade_suf = _clamp01(n_trades / 30.0)
    cap_deg = capacity.get("degradation", {}) if capacity else {}
    cap_score = _clamp01(1.0 - float(cap_deg.get("sharpe_degradation", 0.0) or 0.0) / 3.0)
    # statistical confidence: prefer deflated sharpe / adjusted significance if present
    if "deflated_sharpe" in mt:
        stat = _norm_sharpe(float(mt.get("deflated_sharpe") or 0.0))
    elif "confidence" in mt:
        stat = _clamp01(float(mt["confidence"]))
    else:
        # mild penalty for many trials
        n_cfg = float(mt.get("n_configurations_tested", 1) or 1)
        stat = _clamp01(1.0 / np_log1p(n_cfg))

    components = {
        "net_sharpe": _norm_sharpe(net_sharpe),
        "net_expectancy": _clamp01(0.5 + expectancy * 10.0),
        "drawdown": 1.0 - _norm_dd(dd),
        "stability": _clamp01(stability),
        "oos": oos_score,
        "turnover": 1.0 - _norm_turn(ann_to),
        "costs": 1.0 - cost_ratio,
        "trade_count": trade_suf,
        "capacity": cap_score,
        "statistical_confidence": stat,
    }
    score = sum(w.get(k, 0.0) * components[k] for k in components)
    return {
        "score": float(score),
        "components": components,
        "weights": w,
        "formula_doc": __doc__,
    }


def np_log1p(x: float) -> float:
    import math

    return math.log1p(max(float(x), 0.0))


def classify_horizon(
    metrics: Mapping[str, Any],
    *,
    oos: Mapping[str, Any] | None = None,
    costs: Mapping[str, Any] | None = None,
    neighborhood: Mapping[str, Any] | None = None,
    available: bool = True,
    insufficient: bool = False,
    gates: Mapping[str, Any] | None = None,
) -> tuple[HorizonStatus, str]:
    """Assign ROBUST / PROMISING / FRAGILE / … classification."""
    if not available:
        return HorizonStatus.UNAVAILABLE, "requested frequency finer than native dataset"
    if insufficient:
        return HorizonStatus.INSUFFICIENT_DATA, "insufficient bars/trades for evaluation"

    g = {**DEFAULT_ROBUST_GATES, **dict(gates or {})}
    oos = dict(oos or {})
    costs = dict(costs or {})
    neighborhood = dict(neighborhood or {})

    net_sharpe = float(metrics.get("net_sharpe", metrics.get("sharpe", 0.0)) or 0.0)
    gross_sharpe = float(metrics.get("gross_sharpe", 0.0) or 0.0)
    dd = float(metrics.get("maximum_drawdown", 0.0) or 0.0)
    n_trades = int(metrics.get("trade_count", 0) or 0)
    expectancy = float(metrics.get("expectancy_per_trade", 0.0) or 0.0)
    oos_sharpe = float(oos.get("net_sharpe", oos.get("sharpe", net_sharpe)) or 0.0)
    oos_exp = float(oos.get("expectancy_per_trade", expectancy) or 0.0)
    cost_ineff = bool(costs.get("cost_eroded_edge")) or bool(
        gross_sharpe >= 1.0 and net_sharpe < 0.5
    )
    fragile = bool(neighborhood.get("fragile", False))
    stab = float(neighborhood.get("stability_score", 1.0) or 1.0)

    if cost_ineff:
        return HorizonStatus.COST_INEFFICIENT, "edge collapses after realistic costs"
    if oos and oos_sharpe < float(g["min_oos_sharpe"]) and oos.get("evaluated"):
        return HorizonStatus.OOS_FAILURE, "out-of-sample Sharpe below threshold"
    if oos and oos.get("evaluated") and oos_exp < float(g["min_oos_expectancy"]):
        return HorizonStatus.OOS_FAILURE, "out-of-sample expectancy not positive"
    if n_trades < int(g["min_trades"]):
        return HorizonStatus.INSUFFICIENT_DATA, f"trade_count {n_trades} < min_trades"
    if fragile or stab < float(g["min_neighborhood_stability"]):
        return HorizonStatus.FRAGILE, "performance isolated to a narrow horizon neighborhood"
    if dd > float(g["max_drawdown"]):
        return HorizonStatus.FRAGILE, "drawdown exceeds configured maximum"

    robust_ok = (
        oos_exp >= float(g["min_oos_expectancy"])
        and oos_sharpe >= float(g["min_oos_sharpe"])
        and dd <= float(g["max_drawdown"])
        and n_trades >= int(g["min_trades"])
        and stab >= float(g["min_neighborhood_stability"])
        and (not g["require_positive_net_expectancy"] or expectancy > 0)
        and net_sharpe > 0
    )
    if robust_ok and (not oos or oos.get("evaluated", True)):
        return HorizonStatus.ROBUST, "passes configurable robust-horizon gates"
    if net_sharpe > 0 and expectancy >= 0:
        return HorizonStatus.PROMISING, "positive net metrics but not fully robust"
    return HorizonStatus.FRAGILE, "does not meet promising/robust criteria"


def select_best_robust_horizon(
    results: Sequence[HorizonResult],
    *,
    gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select BEST ROBUST HORIZON (gated), distinct from best in-sample score."""
    scored = [r for r in results if r.status != HorizonStatus.UNAVAILABLE]
    in_sample = max(
        scored,
        key=lambda r: float(r.robustness_score or r.metrics.get("net_sharpe", -1e9) or -1e9),
        default=None,
    )
    robust = [
        r
        for r in scored
        if r.status == HorizonStatus.ROBUST
        or classify_horizon(
            r.metrics,
            oos=r.oos,
            costs=r.costs,
            neighborhood=r.neighborhood,
            gates=gates,
        )[0]
        == HorizonStatus.ROBUST
    ]
    best_robust = max(
        robust,
        key=lambda r: float(r.robustness_score or -1e9),
        default=None,
    )
    return {
        "best_robust_horizon": best_robust.to_dict() if best_robust else None,
        "best_in_sample_horizon": in_sample.to_dict() if in_sample else None,
        "are_identical": (
            best_robust is not None
            and in_sample is not None
            and best_robust.spec.key == in_sample.spec.key
        ),
        "note": (
            "BEST ROBUST HORIZON ≠ BEST IN-SAMPLE HORIZON. "
            "Selection requires OOS / cost / neighborhood gates."
        ),
    }


__all__ = [
    "DEFAULT_ROBUST_GATES",
    "DEFAULT_SCORE_WEIGHTS",
    "HorizonScoreWeights",
    "classify_horizon",
    "compute_horizon_research_score",
    "select_best_robust_horizon",
]
