"""Aggressor trade model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from iqrp.app.data.models._timestamps import ensure_utc_ms


class Trade(BaseModel):
    """Aggressor trade print."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    trade_id: str
    timestamp: datetime
    price: float
    size: float
    side: str | None = None
    is_buyer_maker: bool | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> datetime:
        return ensure_utc_ms(value)

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "price": self.price,
            "size": self.size,
            "side": self.side,
            "is_buyer_maker": self.is_buyer_maker,
        }
