"""Signal retirement / degradation recommendations."""

from __future__ import annotations

from typing import Any, Mapping


def evaluate_retirement(
    *,
    ic_recent: float | None = None,
    ic_baseline: float | None = None,
    net_sharpe: float | None = None,
    gross_sharpe: float | None = None,
    cost_ratio: float | None = None,
    capacity: float | None = None,
    capacity_baseline: float | None = None,
    drift_severity: str | None = None,
    regime_unstable: bool | None = None,
    performance_decayed: bool | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Recommend ACTIVE / DEGRADED / RETIRED with explicit reasons.

    Thresholds (defaults):
    - ``ic_collapse_ratio``: recent |IC| / baseline |IC| below this → collapse
    - ``ic_degrade_ratio``: softer IC degradation
    - ``net_sharpe_retire``: net Sharpe below this → retire candidate
    - ``cost_dominance``: costs / gross edge above this
    - ``capacity_collapse_ratio``: capacity / baseline below this
    """
    thr = {
        "ic_collapse_ratio": 0.25,
        "ic_degrade_ratio": 0.50,
        "net_sharpe_retire": 0.0,
        "net_sharpe_degrade": 0.3,
        "cost_dominance": 0.80,
        "capacity_collapse_ratio": 0.30,
        "capacity_degrade_ratio": 0.60,
    }
    if thresholds:
        thr.update({k: float(v) for k, v in thresholds.items()})

    reasons: list[str] = []
    retire_votes = 0
    degrade_votes = 0

    # IC collapse
    if ic_recent is not None and ic_baseline is not None:
        base = abs(float(ic_baseline))
        recent = float(ic_recent)
        if base > 1e-9:
            ratio = abs(recent) / base
            sign_flip = (recent * float(ic_baseline)) < 0 and abs(recent) > 1e-8
            if ratio <= thr["ic_collapse_ratio"] or sign_flip:
                reasons.append("ic_collapse")
                retire_votes += 1
            elif ratio <= thr["ic_degrade_ratio"]:
                reasons.append("ic_degradation")
                degrade_votes += 1
        elif abs(recent) < 1e-8 and base <= 1e-9:
            # both near zero — treat recent zero vs positive baseline handled above;
            # if baseline ~0 but we still have non-positive recent with negative sharpe, other rules apply
            pass

    # Explicit: ic_recent ~ 0 with positive baseline
    if (
        ic_recent is not None
        and ic_baseline is not None
        and abs(float(ic_recent)) < 1e-12
        and abs(float(ic_baseline)) > 1e-6
    ):
        if "ic_collapse" not in reasons:
            reasons.append("ic_collapse")
            retire_votes += 1

    # Cost dominance
    if cost_ratio is not None and float(cost_ratio) >= thr["cost_dominance"]:
        reasons.append("cost_dominance")
        retire_votes += 1
    elif (
        gross_sharpe is not None
        and net_sharpe is not None
        and float(gross_sharpe) > 0.5
        and float(net_sharpe) < thr["net_sharpe_retire"]
    ):
        reasons.append("cost_dominance")
        retire_votes += 1

    # Net sharpe
    if net_sharpe is not None:
        ns = float(net_sharpe)
        if ns < thr["net_sharpe_retire"]:
            reasons.append("net_sharpe_negative")
            # negative net sharpe with IC collapse → retire; alone can be degrade/retire
            if ns < -0.25:
                retire_votes += 1
            else:
                degrade_votes += 1
        elif ns < thr["net_sharpe_degrade"]:
            reasons.append("net_sharpe_weak")
            degrade_votes += 1

    # Capacity collapse
    if capacity is not None and capacity_baseline is not None and float(capacity_baseline) > 1e-12:
        cr = float(capacity) / float(capacity_baseline)
        if cr <= thr["capacity_collapse_ratio"]:
            reasons.append("capacity_collapse")
            retire_votes += 1
        elif cr <= thr["capacity_degrade_ratio"]:
            reasons.append("capacity_degradation")
            degrade_votes += 1
    elif capacity is not None and float(capacity) <= 0.05:
        reasons.append("capacity_collapse")
        retire_votes += 1

    # Drift
    if drift_severity in ("high", "critical"):
        reasons.append("model_drift")
        if drift_severity == "critical":
            retire_votes += 1
        else:
            degrade_votes += 1
    elif drift_severity == "medium":
        reasons.append("model_drift")
        degrade_votes += 1

    if regime_unstable:
        reasons.append("regime_instability")
        degrade_votes += 1

    if performance_decayed:
        reasons.append("performance_decay")
        degrade_votes += 1

    # Decision
    if retire_votes >= 2 or (retire_votes >= 1 and ("ic_collapse" in reasons or "cost_dominance" in reasons or "capacity_collapse" in reasons)):
        # Single strong structural failure with negative economics → RETIRED
        if retire_votes >= 1 and (
            "ic_collapse" in reasons
            or "cost_dominance" in reasons
            or "capacity_collapse" in reasons
        ) and (net_sharpe is not None and float(net_sharpe) < thr["net_sharpe_retire"]):
            status = "RETIRED"
        elif retire_votes >= 2:
            status = "RETIRED"
        else:
            status = "DEGRADED"
    elif retire_votes >= 1 or degrade_votes >= 1:
        status = "DEGRADED"
    else:
        status = "ACTIVE"

    # Smoke-test path: ic collapse + negative net sharpe → at least DEGRADED, prefer RETIRED
    if "ic_collapse" in reasons and net_sharpe is not None and float(net_sharpe) < 0:
        status = "RETIRED"

    return {
        "name": "evaluate_retirement",
        "status": status,
        "recommend": status,
        "reasons": reasons,
        "retire_votes": retire_votes,
        "degrade_votes": degrade_votes,
        "metrics": {
            "ic_recent": ic_recent,
            "ic_baseline": ic_baseline,
            "net_sharpe": net_sharpe,
            "gross_sharpe": gross_sharpe,
            "cost_ratio": cost_ratio,
            "capacity": capacity,
            "capacity_baseline": capacity_baseline,
            "drift_severity": drift_severity,
            "regime_unstable": regime_unstable,
            "performance_decayed": performance_decayed,
        },
        "thresholds": thr,
    }


def batch_evaluate_retirement(
    signals: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate many signals; each value is kwargs for ``evaluate_retirement``."""
    return {
        name: evaluate_retirement(thresholds=thresholds, **kwargs)  # type: ignore[arg-type]
        for name, kwargs in signals.items()
    }
