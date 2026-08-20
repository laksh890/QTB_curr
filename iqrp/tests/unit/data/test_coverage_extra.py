"""Additional unit tests to cover adapter and edge-case branches."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import polars as pl
import pytest

from iqrp.app.config.settings import (
    AppSettings,
    DataSettings,
    ExchangeEndpointSettings,
    StorageSettings,
)
from iqrp.app.core.exceptions import DataError, ValidationError
from iqrp.app.data.exchange.adapters.binance import BinanceExchange
from iqrp.app.data.exchange.adapters.bybit import BybitExchange
from iqrp.app.data.exchange.adapters.coinbase import CoinbaseExchange
from iqrp.app.data.exchange.base import BaseExchange
from iqrp.app.data.ingestion.historical import HistoricalIngestor
from iqrp.app.data.ingestion.websocket import WebsocketEngine
from iqrp.app.data.models import Candle
from iqrp.app.data.models._timestamps import ensure_utc_ms
from iqrp.app.data.rate_limiter import AsyncRateLimiter
from iqrp.app.data.services.downloader import DataDownloader
from iqrp.app.data.services.query import MarketDataQueryService
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType, Timeframe
from iqrp.app.data.validation.repair import DataRepair
from iqrp.app.data.validation.validator import DataValidator
from iqrp.tests.unit.data.mock_exchange import MockExchange


@pytest.mark.unit
def test_ensure_utc_ms_variants() -> None:
    assert ensure_utc_ms(datetime(2024, 1, 1, tzinfo=UTC)).year == 2024
    assert ensure_utc_ms(1704067200).year == 2024
    assert ensure_utc_ms("2024-01-01T00:00:00Z").year == 2024
    with pytest.raises(ValidationError):
        ensure_utc_ms(object())  # type: ignore[arg-type]


@pytest.mark.unit
def test_rate_limiter_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        AsyncRateLimiter(0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_base_exchange_http_errors() -> None:
    settings = ExchangeEndpointSettings(
        name="binance",
        rest_base_url="https://api.binance.com",
        ws_base_url="wss://x",
    )
    exchange = BinanceExchange(settings)
    await exchange.open()
    request = httpx.Request("GET", "https://api.binance.com/x")
    response = httpx.Response(500, request=request, text="boom")
    with patch.object(
        exchange.client,
        "request",
        new=AsyncMock(side_effect=httpx.HTTPStatusError("err", request=request, response=response)),
    ):
        with pytest.raises(DataError) as exc:
            await exchange.request_json("GET", "/x")
        assert exc.value.code == "EXCHANGE_HTTP_ERROR"
    with patch.object(
        exchange.client,
        "request",
        new=AsyncMock(side_effect=httpx.ConnectError("nope", request=request)),
    ):
        with pytest.raises(DataError) as exc2:
            await exchange.request_json("GET", "/x")
        assert exc2.value.code == "EXCHANGE_TRANSPORT_ERROR"
    await exchange.close()
    with pytest.raises(DataError):
        _ = exchange.client


@pytest.mark.unit
@pytest.mark.asyncio
async def test_binance_secondary_endpoints() -> None:
    settings = ExchangeEndpointSettings(
        name="binance", rest_base_url="https://api.binance.com", ws_base_url="wss://x"
    )
    ex = BinanceExchange(settings)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)

    with patch.object(
        ex,
        "request_json",
        new=AsyncMock(return_value=[{"a": 1, "T": 1704067200000, "p": "1", "q": "2", "m": True}]),
    ):
        trades = await ex.fetch_trades("BTCUSDT", start=start, end=end, limit=10)
        assert trades[0].side == "sell"

    with patch.object(
        ex,
        "request_json",
        new=AsyncMock(
            return_value={
                "bids": [["1", "2"]],
                "asks": [["3", "4"]],
                "lastUpdateId": 9,
                "E": 1704067200000,
            }
        ),
    ):
        book = await ex.fetch_orderbook("BTCUSDT")
        assert book.sequence == 9

    with patch.object(
        ex,
        "request_json",
        new=AsyncMock(
            return_value=[{"fundingTime": 1704067200000, "fundingRate": "0.01", "markPrice": "100"}]
        ),
    ):
        funding = await ex.fetch_funding("BTCUSDT", start=start, end=end, limit=10)
        assert funding[0].funding_rate == 0.01

    with patch.object(
        ex,
        "request_json",
        new=AsyncMock(
            return_value=[
                {
                    "timestamp": 1704067200000,
                    "sumOpenInterest": "10",
                    "sumOpenInterestValue": "1000",
                }
            ]
        ),
    ):
        oi = await ex.fetch_open_interest("BTCUSDT", start=start, end=end, limit=10)
        assert oi[0].open_interest == 10

    with patch.object(
        ex,
        "request_json",
        new=AsyncMock(
            return_value={
                "time": 1704067200000,
                "markPrice": "100",
                "indexPrice": "99",
            }
        ),
    ):
        mark = await ex.fetch_mark_price("BTCUSDT")
        index = await ex.fetch_index_price("BTCUSDT")
        assert mark.mark_price == 100
        assert index.index_price == 99

    with patch.object(
        ex,
        "request_json",
        new=AsyncMock(
            return_value=[
                {
                    "time": 1704067200000,
                    "side": "SELL",
                    "price": "90",
                    "origQty": "1",
                    "orderId": 1,
                }
            ]
        ),
    ):
        liqs = await ex.fetch_liquidations("BTCUSDT", start=start, end=end, limit=10)
        assert liqs[0].side == "sell"

    with pytest.raises(DataError):
        await ex.fetch_candles("BTCUSDT", "2w", start=start, end=end, limit=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bybit_and_coinbase_secondary() -> None:
    bybit = BybitExchange(
        ExchangeEndpointSettings(
            name="bybit", rest_base_url="https://api.bybit.com", ws_base_url="wss://x"
        )
    )
    coin = CoinbaseExchange(
        ExchangeEndpointSettings(
            name="coinbase",
            rest_base_url="https://api.exchange.coinbase.com",
            ws_base_url="wss://x",
        )
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)

    with patch.object(
        bybit,
        "request_json",
        new=AsyncMock(
            return_value={
                "result": {
                    "list": [
                        {
                            "execId": "1",
                            "time": "1704067200000",
                            "price": "1",
                            "size": "2",
                            "side": "Buy",
                        }
                    ]
                }
            }
        ),
    ):
        assert await bybit.fetch_trades("BTCUSDT", start=start, end=end, limit=1)

    with patch.object(
        bybit,
        "request_json",
        new=AsyncMock(
            return_value={
                "result": {"ts": 1704067200000, "b": [["1", "1"]], "a": [["2", "2"]], "u": 1}
            }
        ),
    ):
        assert (await bybit.fetch_orderbook("BTCUSDT")).sequence == 1

    with patch.object(
        bybit,
        "request_json",
        new=AsyncMock(
            return_value={
                "result": {
                    "list": [{"fundingRateTimestamp": "1704067200000", "fundingRate": "0.1"}]
                }
            }
        ),
    ):
        assert await bybit.fetch_funding("BTCUSDT", start=start, end=end, limit=1)

    with patch.object(
        bybit,
        "request_json",
        new=AsyncMock(
            return_value={"result": {"list": [{"timestamp": "1704067200000", "openInterest": "5"}]}}
        ),
    ):
        assert await bybit.fetch_open_interest("BTCUSDT", start=start, end=end, limit=1)

    with patch.object(
        bybit,
        "request_json",
        new=AsyncMock(return_value={"result": {"list": [{"markPrice": "1", "indexPrice": "1"}]}}),
    ):
        assert (await bybit.fetch_mark_price("BTCUSDT")).mark_price == 1.0
        assert (await bybit.fetch_index_price("BTCUSDT")).index_price == 1.0

    assert await bybit.fetch_liquidations("BTCUSDT", start=start, end=end, limit=1) == []
    assert bybit.websocket_url("BTCUSDT", "kline") == "wss://x"

    with patch.object(
        coin,
        "request_json",
        new=AsyncMock(
            return_value=[
                {
                    "trade_id": 1,
                    "time": "2024-01-01T00:00:00Z",
                    "price": "1",
                    "size": "1",
                    "side": "buy",
                }
            ]
        ),
    ):
        assert await coin.fetch_trades("BTC-USD", start=start, end=end, limit=1)

    with patch.object(
        coin,
        "request_json",
        new=AsyncMock(return_value={"bids": [["1", "1"]], "asks": [["2", "2"]], "sequence": 3}),
    ):
        assert (await coin.fetch_orderbook("BTC-USD")).sequence == 3

    with patch.object(
        coin,
        "request_json",
        new=AsyncMock(return_value={"price": "100"}),
    ):
        assert (await coin.fetch_mark_price("BTCUSD")).mark_price == 100
        assert (await coin.fetch_index_price("BTCUSD")).index_price == 100

    assert await coin.fetch_funding("BTC-USD", start=start, end=end, limit=1) == []
    assert await coin.fetch_open_interest("BTC-USD", start=start, end=end, limit=1) == []
    assert await coin.fetch_liquidations("BTC-USD", start=start, end=end, limit=1) == []
    assert coin.websocket_url("BTC-USD", "matches") == "wss://x"


@pytest.mark.unit
def test_validator_missing_and_empty() -> None:
    _anomalies, report = DataValidator().validate_candles(
        pl.DataFrame({"x": [1]}), timeframe="1m", exchange="m", symbol="S"
    )
    assert report.missing_pct == 100.0
    _anomalies2, report2 = DataValidator().validate_candles(
        pl.DataFrame(
            schema={
                "open_time": pl.Datetime(time_zone="UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        ),
        timeframe="1m",
        exchange="m",
        symbol="S",
    )
    assert report2.row_count == 0
    assert report2.coverage_pct == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_repair_downloads_only_gaps(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "p")
    catalog = DuckDBCatalog(tmp_path / "db.duckdb", store)
    exchange = MockExchange()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    # Write only first 5 candles
    candles = await exchange.fetch_candles(
        "BTCUSDT", "1m", start=start, end=start + timedelta(minutes=4), limit=100
    )
    frame = pl.DataFrame([c.to_row() for c in candles])
    store.write_frame(
        frame,
        data_type=MarketDataType.CANDLE,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe="1m",
        timestamp_column="open_time",
    )
    repair = DataRepair(
        exchange,
        store,
        catalog,
        page_limit=1000,
        max_retries=2,
        retry_delay=0.01,
        retry_backoff=1.0,
    )
    repaired = await repair.repair_candles(
        "BTCUSDT", "1m", start=start, end=start + timedelta(minutes=9)
    )
    assert repaired.height == 10
    # No-op second pass
    again = await repair.repair_candles(
        "BTCUSDT", "1m", start=start, end=start + timedelta(minutes=9)
    )
    assert again.height == 10
    catalog.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_download_many_and_query_sql(tmp_path: Path) -> None:
    settings = AppSettings(
        storage=StorageSettings(
            data_dir=tmp_path,
            duckdb_path=tmp_path / "iqrp.duckdb",
            parquet_dir=tmp_path / "parquet",
            cache_dir=tmp_path / "cache",
        ),
        data=DataSettings(),
    )
    from iqrp.app.data.exchange.exchange_factory import register_exchange

    register_exchange("mock", lambda s: MockExchange(s))
    # Patch factory.create for multi-download workers
    exchange = MockExchange()
    store = ParquetStore(settings.storage.parquet_dir)
    catalog = DuckDBCatalog(settings.storage.duckdb_path, store)
    downloader = DataDownloader(settings, exchange=exchange, store=store, catalog=catalog)

    def fake_create(name: str | None = None) -> BaseExchange:
        del name
        return MockExchange()

    downloader.factory.create = fake_create  # type: ignore[method-assign]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    results = await downloader.download_many(
        symbols=["BTCUSDT"], timeframes=["1m"], start=start, end=end
    )
    assert "BTCUSDT:1m" in results
    service = MarketDataQueryService(settings, store=store, catalog=None)
    frame = service.sql("SELECT 1 AS n")
    assert frame.height == 1
    catalog.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_reconnect_budget() -> None:
    async def handler(msg: dict[str, Any]) -> None:
        del msg

    class Boom:
        async def __aenter__(self) -> Boom:
            raise RuntimeError("fail")

        async def __aexit__(self, *exc: object) -> None:
            return None

    def connect(_url: str) -> Boom:
        return Boom()

    engine = WebsocketEngine(
        url="wss://x",
        handler=handler,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.02,
        connect=connect,
    )
    with pytest.raises(DataError):
        await engine.run(max_reconnects=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_historical_trades_funding_oi() -> None:
    exchange = MockExchange()
    ingestor = HistoricalIngestor(exchange, page_limit=10, max_retries=2, retry_delay=0.01)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    assert await ingestor.download_trades("BTCUSDT", start=start, end=end)
    assert await ingestor.download_funding("BTCUSDT", start=start, end=end)
    assert await ingestor.download_open_interest("BTCUSDT", start=start, end=end)


@pytest.mark.unit
def test_candle_negative_volume_and_open_bounds() -> None:
    with pytest.raises(ValidationError):
        Candle(
            exchange="m",
            symbol="S",
            timeframe=Timeframe.M1,
            open_time=datetime(2024, 1, 1, tzinfo=UTC),
            open=100,
            high=110,
            low=90,
            close=105,
            volume=-1,
        )
    with pytest.raises(ValidationError):
        Candle(
            exchange="m",
            symbol="S",
            timeframe=Timeframe.M1,
            open_time=datetime(2024, 1, 1, tzinfo=UTC),
            open=120,
            high=110,
            low=90,
            close=105,
            volume=1,
        )


@pytest.mark.unit
def test_parquet_empty_and_missing(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    assert (
        store.write_frame(
            pl.DataFrame(),
            data_type=MarketDataType.CANDLE,
            exchange="m",
            symbol="S",
            timeframe="1m",
            timestamp_column="open_time",
        )
        == []
    )
    with pytest.raises(DataError):
        store.write_frame(
            pl.DataFrame({"a": [1]}),
            data_type=MarketDataType.CANDLE,
            exchange="m",
            symbol="S",
            timeframe="1m",
            timestamp_column="open_time",
        )
    assert store.read(MarketDataType.CANDLE, exchange="m", symbol="S", timeframe="1m").is_empty()
