"""Backtest lifecycle states and shared type aliases.

CRITICAL: A backtest that violates point-in-time correctness must transition
to ``INVALIDATED``. Handlers must never observe data after the event clock.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum
from typing import Any


class BacktestState(str, Enum):
    """Lifecycle states for an institutional backtest run."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"
    ARCHIVED = "ARCHIVED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            BacktestState.COMPLETED,
            BacktestState.FAILED,
            BacktestState.INVALIDATED,
            BacktestState.ARCHIVED,
        }

    @property
    def allows_execution(self) -> bool:
        return self in {BacktestState.CREATED, BacktestState.VALIDATING, BacktestState.RUNNING}


# Convenience aliases used across the backtesting platform.
JSONDict = dict[str, Any]
ReadonlyMapping = Mapping[str, Any]
MutableJSON = MutableMapping[str, Any]
SymbolList = Sequence[str]

__all__ = [
    "BacktestState",
    "JSONDict",
    "MutableJSON",
    "ReadonlyMapping",
    "SymbolList",
]
