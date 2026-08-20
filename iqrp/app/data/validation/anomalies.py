"""Anomaly descriptors for market-data validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class AnomalyKind(StrEnum):
    MISSING_CANDLE = "missing_candle"
    DUPLICATE_CANDLE = "duplicate_candle"
    TIMESTAMP_GAP = "timestamp_gap"
    INCORRECT_ORDER = "incorrect_order"
    NEGATIVE_VOLUME = "negative_volume"
    IMPOSSIBLE_OHLC = "impossible_ohlc"
    BAD_TIMESTAMP = "bad_timestamp"
    MISSING_FIELD = "missing_field"


@dataclass(frozen=True, slots=True)
class Anomaly:
    kind: AnomalyKind
    message: str
    timestamp: datetime | None = None
    details: dict[str, object] | None = None
