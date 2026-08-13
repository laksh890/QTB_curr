"""Signal research result types and lifecycle status.

CRITICAL RULES:
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve (SignalStatus.APPROVED requires
  economic hypothesis, validation evidence, and governance — not Sharpe).
- Status transitions must be auditable (from→to, reason, timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class SignalStatus(str, Enum):
    """Lifecycle states for alpha research candidates."""

    CANDIDATE = "CANDIDATE"
    RESEARCHING = "RESEARCHING"
    VALIDATING = "VALIDATING"
    PROVISIONAL = "PROVISIONAL"
    APPROVED = "APPROVED"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"


# Allowed transitions. REJECTED / RETIRED are terminal for promotion paths
# but DEGRADED may retire; REJECTED is preserved in the experiment registry.
_ALLOWED_TRANSITIONS: dict[SignalStatus, frozenset[SignalStatus]] = {
    SignalStatus.CANDIDATE: frozenset(
        {SignalStatus.RESEARCHING, SignalStatus.REJECTED, SignalStatus.RETIRED}
    ),
    SignalStatus.RESEARCHING: frozenset(
        {
            SignalStatus.VALIDATING,
            SignalStatus.REJECTED,
            SignalStatus.CANDIDATE,
            SignalStatus.RETIRED,
        }
    ),
    SignalStatus.VALIDATING: frozenset(
        {
            SignalStatus.PROVISIONAL,
            SignalStatus.REJECTED,
            SignalStatus.RESEARCHING,
            SignalStatus.RETIRED,
        }
    ),
    SignalStatus.PROVISIONAL: frozenset(
        {
            SignalStatus.APPROVED,
            SignalStatus.DEGRADED,
            SignalStatus.REJECTED,
            SignalStatus.VALIDATING,
            SignalStatus.RETIRED,
        }
    ),
    SignalStatus.APPROVED: frozenset(
        {SignalStatus.DEGRADED, SignalStatus.RETIRED, SignalStatus.PROVISIONAL}
    ),
    SignalStatus.DEGRADED: frozenset(
        {
            SignalStatus.PROVISIONAL,
            SignalStatus.RETIRED,
            SignalStatus.REJECTED,
            SignalStatus.VALIDATING,
        }
    ),
    SignalStatus.RETIRED: frozenset(),
    SignalStatus.REJECTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class StatusTransition:
    """Auditable status change record."""

    from_status: SignalStatus
    to_status: SignalStatus
    reason: str
    timestamp: datetime
    actor: str = "system"
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatusTransition:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            timestamp = datetime.now(UTC)
        return cls(
            from_status=SignalStatus(data["from_status"]),
            to_status=SignalStatus(data["to_status"]),
            reason=str(data.get("reason") or ""),
            timestamp=timestamp,
            actor=str(data.get("actor") or "system"),
            extras=dict(data.get("extras") or {}),
        )


def validate_transition(from_status: SignalStatus, to_status: SignalStatus) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        allowed_names = sorted(s.value for s in allowed)
        raise ValueError(
            f"Illegal status transition {from_status.value} → {to_status.value}. "
            f"Allowed: {allowed_names}"
        )


@dataclass(slots=True)
class SignalStatistics:
    """Descriptive statistics of a signal series (not performance approval)."""

    n_obs: int
    n_finite: int
    mean: float
    std: float
    skew: float
    kurtosis: float
    min: float
    max: float
    missing_pct: float
    autocorrelation_lag1: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "n_finite": self.n_finite,
            "mean": self.mean,
            "std": self.std,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "min": self.min,
            "max": self.max,
            "missing_pct": self.missing_pct,
            "autocorrelation_lag1": self.autocorrelation_lag1,
        }


@dataclass(slots=True)
class SignalPerformance:
    """Predictive / research performance metrics.

    NOTE: Historical Sharpe alone cannot approve. These metrics inform
    research triage; they are not a promotion gate by themselves.
    Statistical significance alone ≠ alpha.
    """

    ic_mean: float = float("nan")
    ic_std: float = float("nan")
    rank_ic_mean: float = float("nan")
    hit_rate: float = float("nan")
    predictive_r2: float = float("nan")
    sharpe_proxy: float = float("nan")
    turnover_proxy: float = float("nan")
    n_splits: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "rank_ic_mean": self.rank_ic_mean,
            "hit_rate": self.hit_rate,
            "predictive_r2": self.predictive_r2,
            "sharpe_proxy": self.sharpe_proxy,
            "turnover_proxy": self.turnover_proxy,
            "n_splits": self.n_splits,
            "extras": dict(self.extras),
            "disclaimer": (
                "Historical Sharpe alone cannot approve. "
                "Statistical significance alone ≠ alpha."
            ),
        }


@dataclass(slots=True)
class SignalScore:
    """Composite research score for triage (not approval)."""

    overall: float
    predictive: float
    stability: float
    persistence: float
    economic_hypothesis_score: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "predictive": self.predictive,
            "stability": self.stability,
            "persistence": self.persistence,
            "economic_hypothesis_score": self.economic_hypothesis_score,
            "notes": self.notes,
        }


@dataclass(slots=True)
class SignalResearchReport:
    """Bundle of research outputs for a signal experiment."""

    signal_name: str
    version: str
    status: SignalStatus
    statistics: SignalStatistics | None = None
    performance: SignalPerformance | None = None
    score: SignalScore | None = None
    economic_hypothesis: str = ""
    transitions: list[StatusTransition] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_name": self.signal_name,
            "version": self.version,
            "status": self.status.value,
            "statistics": None if self.statistics is None else self.statistics.to_dict(),
            "performance": None if self.performance is None else self.performance.to_dict(),
            "score": None if self.score is None else self.score.to_dict(),
            "economic_hypothesis": self.economic_hypothesis,
            "transitions": [t.to_dict() for t in self.transitions],
            "diagnostics": dict(self.diagnostics),
            "created_at": self.created_at.isoformat(),
            "warnings": list(self.warnings),
            "rules": {
                "statistical_significance_alone_is_not_alpha": True,
                "historical_sharpe_alone_cannot_approve": True,
                "economic_hypothesis_required": True,
                "point_in_time_no_future_leakage": True,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignalResearchReport:
        created = data.get("created_at")
        if isinstance(created, str):
            created_at = datetime.fromisoformat(created)
        elif isinstance(created, datetime):
            created_at = created
        else:
            created_at = datetime.now(UTC)

        stats_raw = data.get("statistics")
        perf_raw = data.get("performance")
        score_raw = data.get("score")

        statistics = None
        if isinstance(stats_raw, dict):
            statistics = SignalStatistics(
                n_obs=int(stats_raw.get("n_obs", 0)),
                n_finite=int(stats_raw.get("n_finite", 0)),
                mean=float(stats_raw.get("mean", float("nan"))),
                std=float(stats_raw.get("std", float("nan"))),
                skew=float(stats_raw.get("skew", float("nan"))),
                kurtosis=float(stats_raw.get("kurtosis", float("nan"))),
                min=float(stats_raw.get("min", float("nan"))),
                max=float(stats_raw.get("max", float("nan"))),
                missing_pct=float(stats_raw.get("missing_pct", float("nan"))),
                autocorrelation_lag1=float(
                    stats_raw.get("autocorrelation_lag1", float("nan"))
                ),
            )

        performance = None
        if isinstance(perf_raw, dict):
            performance = SignalPerformance(
                ic_mean=float(perf_raw.get("ic_mean", float("nan"))),
                ic_std=float(perf_raw.get("ic_std", float("nan"))),
                rank_ic_mean=float(perf_raw.get("rank_ic_mean", float("nan"))),
                hit_rate=float(perf_raw.get("hit_rate", float("nan"))),
                predictive_r2=float(perf_raw.get("predictive_r2", float("nan"))),
                sharpe_proxy=float(perf_raw.get("sharpe_proxy", float("nan"))),
                turnover_proxy=float(perf_raw.get("turnover_proxy", float("nan"))),
                n_splits=int(perf_raw.get("n_splits", 0)),
                extras=dict(perf_raw.get("extras") or {}),
            )

        score = None
        if isinstance(score_raw, dict):
            score = SignalScore(
                overall=float(score_raw.get("overall", float("nan"))),
                predictive=float(score_raw.get("predictive", float("nan"))),
                stability=float(score_raw.get("stability", float("nan"))),
                persistence=float(score_raw.get("persistence", float("nan"))),
                economic_hypothesis_score=float(
                    score_raw.get("economic_hypothesis_score", float("nan"))
                ),
                notes=str(score_raw.get("notes") or ""),
            )

        return cls(
            signal_name=str(data["signal_name"]),
            version=str(data["version"]),
            status=SignalStatus(data["status"]),
            statistics=statistics,
            performance=performance,
            score=score,
            economic_hypothesis=str(data.get("economic_hypothesis") or ""),
            transitions=[StatusTransition.from_dict(t) for t in (data.get("transitions") or [])],
            diagnostics=dict(data.get("diagnostics") or {}),
            created_at=created_at,
            warnings=list(data.get("warnings") or []),
        )
