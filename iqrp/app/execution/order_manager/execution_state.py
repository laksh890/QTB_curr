"""Execution-level state machine (parent/basket lifecycle).

CRITICAL RULES
--------------
- Execution never generates alpha.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Never override hard risk limits.
"""

from __future__ import annotations

from enum import Enum

from iqrp.app.core.exceptions import ExecutionError


class ExecutionState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    VALIDATING = "VALIDATING"
    EXECUTING = "EXECUTING"
    PARTIALLY_EXECUTED = "PARTIALLY_EXECUTED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    HALTED = "HALTED"


ALLOWED_EXECUTION_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.IDLE: frozenset(
        {ExecutionState.PLANNING, ExecutionState.HALTED, ExecutionState.CANCELLED}
    ),
    ExecutionState.PLANNING: frozenset(
        {
            ExecutionState.VALIDATING,
            ExecutionState.CANCELLED,
            ExecutionState.FAILED,
            ExecutionState.HALTED,
        }
    ),
    ExecutionState.VALIDATING: frozenset(
        {
            ExecutionState.EXECUTING,
            ExecutionState.CANCELLED,
            ExecutionState.FAILED,
            ExecutionState.HALTED,
        }
    ),
    ExecutionState.EXECUTING: frozenset(
        {
            ExecutionState.PARTIALLY_EXECUTED,
            ExecutionState.COMPLETING,
            ExecutionState.COMPLETED,
            ExecutionState.CANCELLED,
            ExecutionState.FAILED,
            ExecutionState.HALTED,
        }
    ),
    ExecutionState.PARTIALLY_EXECUTED: frozenset(
        {
            ExecutionState.EXECUTING,
            ExecutionState.COMPLETING,
            ExecutionState.COMPLETED,
            ExecutionState.CANCELLED,
            ExecutionState.FAILED,
            ExecutionState.HALTED,
        }
    ),
    ExecutionState.COMPLETING: frozenset(
        {
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.HALTED,
        }
    ),
    ExecutionState.COMPLETED: frozenset(),
    ExecutionState.CANCELLED: frozenset(),
    ExecutionState.FAILED: frozenset(),
    ExecutionState.HALTED: frozenset(
        {
            ExecutionState.IDLE,
            ExecutionState.PLANNING,
            ExecutionState.CANCELLED,
            ExecutionState.FAILED,
        }
    ),
}


def can_execution_transition(current: ExecutionState, new_state: ExecutionState) -> bool:
    return new_state in ALLOWED_EXECUTION_TRANSITIONS.get(current, frozenset())


def assert_execution_transition(current: ExecutionState, new_state: ExecutionState) -> None:
    if not can_execution_transition(current, new_state):
        raise ExecutionError(
            f"Illegal execution state transition: {current.value} -> {new_state.value}",
            code="EXECUTION_STATE_TRANSITION_ILLEGAL",
            details={"from": current.value, "to": new_state.value},
        )


def transition_execution(current: ExecutionState, new_state: ExecutionState) -> ExecutionState:
    assert_execution_transition(current, new_state)
    return new_state
