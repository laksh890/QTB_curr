"""Horizon research types and classifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HorizonStatus(str, Enum):
    ROBUST = "ROBUST"
    PROMISING = "PROMISING"
    FRAGILE = "FRAGILE"
    COST_INEFFICIENT = "COST-INEFFICIENT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    OOS_FAILURE = "OOS_FAILURE"
    UNAVAILABLE = "UNAVAILABLE"


class SignalSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Timeframe:
    """Canonical timeframe token, e.g. 1m / 5m / 1h / 1D."""

    label: str
    seconds: float

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class HoldingPeriod:
    """Holding as N bars and/or wall-clock seconds."""

    bars: int | None = None
    seconds: float | None = None
    label: str = ""

    def __str__(self) -> str:
        if self.label:
            return self.label
        if self.bars is not None:
            return f"{self.bars}bar"
        if self.seconds is not None:
            return f"{int(self.seconds)}s"
        return "unspecified"


@dataclass
class HorizonSpec:
    """Separates data timeframe, signal timeframe, and holding period."""

    data_timeframe: Timeframe
    signal_timeframe: Timeframe
    holding: HoldingPeriod
    instrument: str = ""
    strategy_id: str = ""
    strategy_version: str = "1.0.0"

    @property
    def key(self) -> str:
        return (
            f"{self.strategy_id}|{self.instrument}|"
            f"{self.data_timeframe}|{self.signal_timeframe}|{self.holding}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_timeframe": str(self.data_timeframe),
            "signal_timeframe": str(self.signal_timeframe),
            "holding": str(self.holding),
            "holding_bars": self.holding.bars,
            "holding_seconds": self.holding.seconds,
            "instrument": self.instrument,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "key": self.key,
        }


@dataclass
class HorizonResult:
    """One cell of the horizon research matrix."""

    spec: HorizonSpec
    status: HorizonStatus
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    trade_frequency: dict[str, Any] = field(default_factory=dict)
    costs: dict[str, Any] = field(default_factory=dict)
    turnover: dict[str, Any] = field(default_factory=dict)
    capacity: dict[str, Any] = field(default_factory=dict)
    half_life: dict[str, Any] = field(default_factory=dict)
    oos: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    overtrading: dict[str, Any] = field(default_factory=dict)
    robustness_score: float | None = None
    neighborhood: dict[str, Any] = field(default_factory=dict)
    multiple_testing: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Research / simulated result only. Not live performance. "
        "Capacity estimates are ESTIMATED / MODEL-BASED when present."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "trade_frequency": dict(self.trade_frequency),
            "costs": dict(self.costs),
            "turnover": dict(self.turnover),
            "capacity": dict(self.capacity),
            "half_life": dict(self.half_life),
            "oos": dict(self.oos),
            "regime": dict(self.regime),
            "overtrading": dict(self.overtrading),
            "robustness_score": self.robustness_score,
            "neighborhood": dict(self.neighborhood),
            "multiple_testing": dict(self.multiple_testing),
            "disclaimer": self.disclaimer,
        }


DEFAULT_DATA_TIMEFRAMES: tuple[str, ...] = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1D",
)

DEFAULT_HOLDING_BARS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40)

DEFAULT_CAPITAL_LEVELS: tuple[float, ...] = (
    100_000.0,
    500_000.0,
    1_000_000.0,
    5_000_000.0,
    10_000_000.0,
    50_000_000.0,
)


__all__ = [
    "DEFAULT_CAPITAL_LEVELS",
    "DEFAULT_DATA_TIMEFRAMES",
    "DEFAULT_HOLDING_BARS",
    "HoldingPeriod",
    "HorizonResult",
    "HorizonSpec",
    "HorizonStatus",
    "SignalSide",
    "Timeframe",
]
