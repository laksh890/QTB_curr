"""Unit tests for market-data models and types."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.data.models import Candle, Trade
from iqrp.app.data.types import Timeframe, timeframe_to_ms, utc_to_ms


@pytest.mark.unit
def test_candle_valid() -> None:
    c = Candle(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe=Timeframe.M1,
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        open=100,
        high=110,
        low=90,
        close=105,
        volume=1.5,
    )
    assert c.to_row()["symbol"] == "BTCUSDT"
    assert utc_to_ms(c.open_time) == 1704067200000


@pytest.mark.unit
def test_candle_rejects_impossible_ohlc() -> None:
    with pytest.raises(ValidationError):
        Candle(
            exchange="binance",
            symbol="BTCUSDT",
            timeframe=Timeframe.M1,
            open_time=datetime(2024, 1, 1, tzinfo=UTC),
            open=100,
            high=90,
            low=95,
            close=100,
            volume=1,
        )


@pytest.mark.unit
def test_trade_parses_ms_timestamp() -> None:
    t = Trade(
        exchange="binance",
        symbol="BTCUSDT",
        trade_id="1",
        timestamp=1704067200000,  # type: ignore[arg-type]
        price=1.0,
        size=1.0,
    )
    assert t.timestamp.tzinfo is not None


@pytest.mark.unit
def test_timeframe_ms() -> None:
    assert timeframe_to_ms("1m") == 60_000
    assert timeframe_to_ms(Timeframe.H1) == 3_600_000
