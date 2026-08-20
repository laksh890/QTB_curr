"""Order book snapshot models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from iqrp.app.data.models._timestamps import ensure_utc_ms


class OrderBookLevel(BaseModel):
    """Single price level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    price: float
    size: float


class OrderBook(BaseModel):
    """Order book snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timestamp: datetime
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    sequence: int | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> datetime:
        return ensure_utc_ms(value)

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "best_bid": self.bids[0].price if self.bids else None,
            "best_ask": self.asks[0].price if self.asks else None,
            "bid_size": self.bids[0].size if self.bids else None,
            "ask_size": self.asks[0].size if self.asks else None,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
        }
