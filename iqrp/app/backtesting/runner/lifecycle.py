"""Auditable operational runner lifecycle (distinct from EventDrivenEngine BacktestState)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from iqrp.app.backtesting.types import BacktestState


class RunnerLifecycleState(str, Enum):
    """Operational runner states (may be richer than engine BacktestState)."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INVALIDATED = "INVALIDATED"
    ARCHIVED = "ARCHIVED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunnerLifecycleState.COMPLETED,
            RunnerLifecycleState.FAILED,
            RunnerLifecycleState.CANCELLED,
            RunnerLifecycleState.INVALIDATED,
            RunnerLifecycleState.ARCHIVED,
        }


_ENGINE_TO_RUNNER: dict[BacktestState, RunnerLifecycleState] = {
    BacktestState.CREATED: RunnerLifecycleState.CREATED,
    BacktestState.VALIDATING: RunnerLifecycleState.VALIDATING,
    BacktestState.RUNNING: RunnerLifecycleState.RUNNING,
    BacktestState.COMPLETED: RunnerLifecycleState.COMPLETED,
    BacktestState.FAILED: RunnerLifecycleState.FAILED,
    BacktestState.INVALIDATED: RunnerLifecycleState.INVALIDATED,
    BacktestState.ARCHIVED: RunnerLifecycleState.ARCHIVED,
}


def map_engine_state(state: BacktestState | str | None) -> RunnerLifecycleState | None:
    """Map EventDrivenEngine / platform BacktestState → runner lifecycle."""
    if state is None:
        return None
    if isinstance(state, RunnerLifecycleState):
        return state
    if isinstance(state, BacktestState):
        return _ENGINE_TO_RUNNER.get(state)
    try:
        eng = BacktestState(str(state))
    except Exception:  # noqa: BLE001
        try:
            return RunnerLifecycleState(str(state))
        except Exception:  # noqa: BLE001
            return None
    return _ENGINE_TO_RUNNER.get(eng)


def map_runner_to_engine(state: RunnerLifecycleState) -> BacktestState:
    """Map runner lifecycle into the closest existing BacktestState."""
    if state is RunnerLifecycleState.PREPARING:
        return BacktestState.VALIDATING
    if state is RunnerLifecycleState.PAUSED:
        return BacktestState.RUNNING
    if state is RunnerLifecycleState.CANCELLED:
        return BacktestState.FAILED
    try:
        return BacktestState(state.value)
    except Exception:  # noqa: BLE001
        return BacktestState.FAILED


@dataclass
class LifecycleTransition:
    from_state: str
    to_state: str
    timestamp: str
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "meta": dict(self.meta),
        }


@dataclass
class Lifecycle:
    """Timestamped auditable state machine for BacktestRunner."""

    state: RunnerLifecycleState = RunnerLifecycleState.CREATED
    history: list[LifecycleTransition] = field(default_factory=list)

    def transition(
        self,
        to: RunnerLifecycleState,
        *,
        reason: str = "",
        meta: dict[str, Any] | None = None,
        allow_same: bool = False,
    ) -> RunnerLifecycleState:
        if to is self.state and not allow_same:
            return self.state
        if self.state.is_terminal and to is not RunnerLifecycleState.ARCHIVED:
            raise RuntimeError(
                f"cannot transition from terminal state {self.state.value} to {to.value}"
            )
        ts = datetime.now(tz=UTC).isoformat()
        self.history.append(
            LifecycleTransition(
                from_state=self.state.value,
                to_state=to.value,
                timestamp=ts,
                reason=reason,
                meta=dict(meta or {}),
            )
        )
        self.state = to
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "history": [h.to_dict() for h in self.history],
        }


__all__ = [
    "Lifecycle",
    "LifecycleTransition",
    "RunnerLifecycleState",
    "map_engine_state",
    "map_runner_to_engine",
]
