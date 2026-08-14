"""Promotion / validation gates for institutional backtests.

CRITICAL
--------
Never promote a strategy on highest historical return or Sharpe alone.
Out-of-sample evidence is mandatory. Gates cover OOS, risk, drawdown,
capacity, costs, stability, regime robustness, and statistical checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from iqrp.app.backtesting.performance.scorecard import StrategyScorecard

__all__ = [
    "GateResult",
    "GateThresholds",
    "evaluate_gates",
    "require_oos",
]


@dataclass
class GateThresholds:
    """Configurable promotion thresholds.

    ``require_out_of_sample`` defaults to True — promotion without OOS fails.
    Historical Sharpe / return alone is never sufficient.
    """

    require_out_of_sample: bool = True
    min_oos_sharpe: float | None = 0.0
    min_sharpe: float | None = None
    max_drawdown: float | None = 0.35
    max_cvar: float | None = None
    min_stability: float | None = None
    min_regime_robustness: float | None = None
    max_turnover: float | None = None
    max_transaction_costs: float | None = None
    min_capacity: float | None = None
    # Statistical: reject if only IS sharpe is cited without OOS
    reject_in_sample_only: bool = True
    min_statistical_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> GateThresholds:
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class GateResult:
    """Outcome of :func:`evaluate_gates`."""

    approved: bool
    out_of_sample_ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    scorecard: dict[str, Any] = field(default_factory=dict)
    policy: str = (
        "Never promote on highest historical return/Sharpe alone; "
        "out-of-sample evidence is mandatory."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_oos(scorecard: StrategyScorecard | Mapping[str, Any]) -> bool:
    """True iff an out-of-sample metric is present and finite."""
    if isinstance(scorecard, StrategyScorecard):
        oos = scorecard.out_of_sample
    else:
        oos = scorecard.get("out_of_sample")
    if oos is None:
        return False
    try:
        return float(oos) == float(oos)  # NaN check
    except (TypeError, ValueError):
        return False


def evaluate_gates(
    scorecard: StrategyScorecard | Mapping[str, Any],
    gates: GateThresholds | Mapping[str, Any] | None = None,
    *,
    in_sample_sharpe: float | None = None,
    statistical_ok: bool | None = None,
) -> GateResult:
    """Evaluate promotion gates against a strategy scorecard.

    Always fails when OOS is missing if ``require_out_of_sample`` is True.
    Never approves solely because in-sample Sharpe/return is high.
    """
    thr = gates if isinstance(gates, GateThresholds) else GateThresholds.from_dict(gates)
    sc = (
        scorecard
        if isinstance(scorecard, StrategyScorecard)
        else StrategyScorecard.from_dict(scorecard)
    )

    checks: dict[str, bool] = {}
    reasons: list[str] = []

    oos_present = require_oos(sc)
    checks["out_of_sample_present"] = oos_present
    if thr.require_out_of_sample and not oos_present:
        reasons.append("out-of-sample metric missing — cannot promote")

    oos_ok = oos_present
    if thr.min_oos_sharpe is not None and oos_present:
        oos_ok = float(sc.out_of_sample) >= float(thr.min_oos_sharpe)  # type: ignore[arg-type]
        checks["out_of_sample_sharpe"] = oos_ok
        if not oos_ok:
            reasons.append(f"OOS Sharpe {sc.out_of_sample} < min_oos_sharpe {thr.min_oos_sharpe}")
    checks["out_of_sample_ok"] = oos_ok and (not thr.require_out_of_sample or oos_present)

    if thr.reject_in_sample_only:
        # Explicit policy: high IS Sharpe without OOS is not enough
        is_only = (in_sample_sharpe is not None and float(in_sample_sharpe) > 0) and not oos_present
        checks["not_in_sample_only"] = not is_only
        if is_only:
            reasons.append("rejected: in-sample Sharpe alone is insufficient for promotion")

    if thr.min_sharpe is not None:
        checks["sharpe"] = sc.sharpe >= float(thr.min_sharpe)
        if not checks["sharpe"]:
            reasons.append(f"Sharpe {sc.sharpe} < min_sharpe {thr.min_sharpe}")

    if thr.max_drawdown is not None:
        checks["max_drawdown"] = sc.max_drawdown <= float(thr.max_drawdown)
        if not checks["max_drawdown"]:
            reasons.append(f"max_drawdown {sc.max_drawdown} > max_drawdown {thr.max_drawdown}")

    if thr.max_cvar is not None:
        checks["cvar"] = sc.cvar <= float(thr.max_cvar)
        if not checks["cvar"]:
            reasons.append(f"CVaR {sc.cvar} > max_cvar {thr.max_cvar}")

    if thr.min_stability is not None:
        checks["stability"] = sc.stability >= float(thr.min_stability)
        if not checks["stability"]:
            reasons.append(f"stability {sc.stability} < min_stability {thr.min_stability}")

    if thr.min_regime_robustness is not None:
        ok = sc.regime_robustness is not None and float(sc.regime_robustness) >= float(
            thr.min_regime_robustness
        )
        checks["regime_robustness"] = ok
        if not ok:
            reasons.append("regime robustness gate failed")

    if thr.max_turnover is not None:
        checks["turnover"] = sc.turnover <= float(thr.max_turnover)
        if not checks["turnover"]:
            reasons.append(f"turnover {sc.turnover} > max_turnover {thr.max_turnover}")

    if thr.max_transaction_costs is not None:
        checks["transaction_costs"] = sc.transaction_costs <= float(thr.max_transaction_costs)
        if not checks["transaction_costs"]:
            reasons.append("transaction cost gate failed")

    if thr.min_capacity is not None:
        ok = sc.capacity is not None and float(sc.capacity) >= float(thr.min_capacity)
        checks["capacity"] = ok
        if not ok:
            reasons.append("capacity gate failed")

    if statistical_ok is not None:
        checks["statistical"] = bool(statistical_ok)
        if not statistical_ok:
            reasons.append("statistical validation gate failed")
    elif thr.min_statistical_confidence is not None:
        # Without an explicit flag, treat missing statistical evidence as fail
        checks["statistical"] = False
        reasons.append("statistical confidence not provided")

    # Mandatory OOS
    if thr.require_out_of_sample and not checks.get("out_of_sample_ok", False):
        approved = False
    else:
        approved = all(checks.values()) if checks else False
        if not checks:
            approved = False
            reasons.append("no gates evaluated")

    # Final policy guard: never approve without OOS when required
    if thr.require_out_of_sample and not oos_present:
        approved = False

    return GateResult(
        approved=bool(approved),
        out_of_sample_ok=bool(checks.get("out_of_sample_ok", False)),
        checks=checks,
        reasons=reasons,
        scorecard=sc.to_dict(),
    )


def summarize_gate_policy() -> Sequence[str]:
    return (
        "Out-of-sample evidence is mandatory for promotion.",
        "Never promote on highest historical return or Sharpe alone.",
        "Gates include risk, drawdown, capacity, costs, stability, regime, and statistics.",
        "Invalidated / leakage-contaminated experiments cannot be promoted.",
    )
