"""Unit tests for exchange factory and adapter parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iqrp.app.config.settings import DataSettings, ExchangeEndpointSettings
from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.data.exchange.adapters.binance import BinanceExchange
from iqrp.app.data.exchange.adapters.bybit import BybitExchange
from iqrp.app.data.exchange.adapters.coinbase import CoinbaseExchange
from iqrp.app.data.exchange.exchange_factory import (
    ExchangeFactory,
    available_exchanges,
    register_exchange,
)
from iqrp.app.data.types import Timeframe


@pytest.mark.unit
def test_factory_creates_configured_exchanges() -> None:
    factory = ExchangeFactory(DataSettings())
    assert set(available_exchanges()) >= {"binance", "bybit", "coinbase"}
    assert factory.create("binance").name == "binance"
    assert factory.create("bybit").name == "bybit"
    assert factory.create("coinbase").name == "coinbase"


@pytest.mark.unit
def test_factory_unknown_exchange() -> None:
    factory = ExchangeFactory(DataSettings())
    with pytest.raises(ConfigurationError):
        factory.create("unknown")


@pytest.mark.unit
def test_register_custom_exchange() -> None:
    def builder(settings: ExchangeEndpointSettings) -> BinanceExchange:
        return BinanceExchange(settings)

    register_exchange("binance", builder)
    factory = ExchangeFactory(DataSettings())
    assert isinstance(factory.create("binance"), BinanceExchange)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_binance_parse_candles() -> None:
    settings = ExchangeEndpointSettings(
        name="binance",
        rest_base_url="https://api.binance.com",
        ws_base_url="wss://stream.binance.com:9443/ws",
    )
    exchange = BinanceExchange(settings)
    row: list[Any] = [
        1704067200000,
        "100",
        "110",
        "90",
        "105",
        "1.5",
        1704067259999,
        "150",
        10,
        "0",
        "0",
        "0",
    ]
    with patch.object(exchange, "request_json", new=AsyncMock(return_value=[row])):
        candles = await exchange.fetch_candles(
            "BTCUSDT",
            Timeframe.M1,
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            limit=1000,
        )
    assert len(candles) == 1
    assert candles[0].open == 100.0
    assert "btcusdt@kline_1m" in exchange.websocket_url("BTCUSDT", "kline_1m")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bybit_parse_candles() -> None:
    settings = ExchangeEndpointSettings(
        name="bybit",
        rest_base_url="https://api.bybit.com",
        ws_base_url="wss://stream.bybit.com/v5/public/spot",
    )
    exchange = BybitExchange(settings)
    payload = {
        "result": {
            "list": [
                ["1704067200000", "100", "110", "90", "105", "1.5", "150"],
            ]
        }
    }
    with patch.object(exchange, "request_json", new=AsyncMock(return_value=payload)):
        candles = await exchange.fetch_candles(
            "BTCUSDT",
            "1m",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            limit=1000,
        )
    assert candles[0].close == 105.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_coinbase_parse_candles() -> None:
    settings = ExchangeEndpointSettings(
        name="coinbase",
        rest_base_url="https://api.exchange.coinbase.com",
        ws_base_url="wss://ws-feed.exchange.coinbase.com",
    )
    exchange = CoinbaseExchange(settings)
    # [time, low, high, open, close, volume]
    payload = [[1704067200, 90, 110, 100, 105, 1.5]]
    with patch.object(exchange, "request_json", new=AsyncMock(return_value=payload)):
        candles = await exchange.fetch_candles(
            "BTC-USD",
            "1m",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 0, 1, tzinfo=UTC),
            limit=300,
        )
    assert exchange.normalize_symbol("BTCUSDT") == "BTC-USDT"
    assert candles[0].symbol == "BTC-USD"
