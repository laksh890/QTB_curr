"""Funding, open interest, mark/index price, and liquidation models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iqrp.app.data.models._timestamps import ensure_utc_ms


class FundingRate(BaseModel):
    """Perpetual funding rate observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timestamp: datetime
    funding_rate: float
    mark_price: float | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> datetime:
        return ensure_utc_ms(value)

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "funding_rate": self.funding_rate,
            "mark_price": self.mark_price,
        }


class OpenInterest(BaseModel):
    """Open interest observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timestamp: datetime
    open_interest: float
    open_interest_value: float | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> datetime:
        return ensure_utc_ms(value)

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "open_interest": self.open_interest,
            "open_interest_value": self.open_interest_value,
        }


class MarkPrice(BaseModel):
    """Mark price observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timestamp: datetime
    mark_price: float
    index_price: float | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> datetime:
        return ensure_utc_ms(value)

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
        }


class IndexPrice(BaseModel):
    """Index price observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timestamp: datetime
    index_price: float

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> datetime:
        return ensure_utc_ms(value)

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "index_price": self.index_price,
        }


class Liquidation(BaseModel):
    """Forced liquidation event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timestamp: datetime
    side: str
    price: float
    size: float
    order_id: str | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> datetime:
        return ensure_utc_ms(value)

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "side": self.side,
            "price": self.price,
            "size": self.size,
            "order_id": self.order_id,
        }


class DataQualityReport(BaseModel):
    """Aggregate data-quality metrics for a stored series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timeframe: str | None = None
    data_type: str
    row_count: int = 0
    missing_pct: float = 0.0
    duplicate_count: int = 0
    gap_count: int = 0
    coverage_pct: float = 0.0
    oldest_record: datetime | None = None
    newest_record: datetime | None = None
    exchange_latency_ms: float | None = None
    issues: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("oldest_record", "newest_record", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> Any:
        if value is None:
            return None
        return ensure_utc_ms(value)
