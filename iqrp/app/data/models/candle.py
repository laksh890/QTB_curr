"""OHLCV candle model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.data.models._timestamps import ensure_utc_ms
from iqrp.app.data.types import Timeframe


class Candle(BaseModel):
    """OHLCV candle with millisecond-precision UTC open time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exchange: str
    symbol: str
    timeframe: Timeframe
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime | None = None
    quote_volume: float | None = None
    trade_count: int | None = None

    @field_validator("open_time", "close_time", mode="before")
    @classmethod
    def _parse_ts(cls, value: Any) -> Any:
        if value is None:
            return None
        return ensure_utc_ms(value)

    @model_validator(mode="after")
    def _validate_ohlc(self) -> Candle:
        if self.volume < 0:
            raise ValidationError("Negative volume", code="NEGATIVE_VOLUME")
        if self.high < self.low:
            raise ValidationError("High < low", code="IMPOSSIBLE_OHLC")
        if self.open > self.high or self.open < self.low:
            raise ValidationError("Open outside high/low", code="IMPOSSIBLE_OHLC")
        if self.close > self.high or self.close < self.low:
            raise ValidationError("Close outside high/low", code="IMPOSSIBLE_OHLC")
        return self

    def to_row(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "open_time": self.open_time,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time": self.close_time,
            "quote_volume": self.quote_volume,
            "trade_count": self.trade_count,
        }
