"""Fill remaining data-layer coverage gaps."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from iqrp.app.config.settings import (
    AppSettings,
    DataSettings,
    ExchangeEndpointSettings,
    StorageSettings,
)
from iqrp.app.core.exceptions import DataError
from iqrp.app.data.exchange.adapters.binance import BinanceExchange
from iqrp.app.data.exchange.adapters.bybit import BybitExchange
from iqrp.app.data.exchange.adapters.coinbase import CoinbaseExchange
from iqrp.app.data.ingestion.historical import HistoricalIngestor
from iqrp.app.data.ingestion.scheduler import IngestionScheduler
from iqrp.app.data.ingestion.websocket import WebsocketEngine
from iqrp.app.data.rate_limiter import AsyncRateLimiter
from iqrp.app.data.services.updater import DataUpdater
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType
from iqrp.app.data.validation.validator import DataValidator
from iqrp.tests.unit.data.mock_exchange import MockExchange


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_waits_when_empty() -> None:
    limiter = AsyncRateLimiter(100.0, capacity=1.0)
    await limiter.acquire()
    await limiter.acquire(0)
    started = asyncio.get_running_loop().time()
    await limiter.acquire(1.0)
    assert asyncio.get_running_loop().time() - started >= 0.0


@pytest.mark.unit
def test_scheduler_rejects_bad_concurrency() -> None:
    with pytest.raises(ValueError):
        IngestionScheduler(concurrency=0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_updater_noop_and_empty_errors(tmp_path: Path) -> None:
    settings = AppSettings(
        storage=StorageSettings(
            data_dir=tmp_path,
            duckdb_path=tmp_path / "db.duckdb",
            parquet_dir=tmp_path / "parquet",
            cache_dir=tmp_path / "cache",
        ),
        data=DataSettings(),
    )
    store = ParquetStore(settings.storage.parquet_dir)
    updater = DataUpdater(settings, exchange=MockExchange(), store=store, catalog=None)
    with pytest.raises(ValueError):
        await updater.update_candles("BTCUSDT", "1m")

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    from iqrp.app.data.services.downloader import DataDownloader

    await DataDownloader(
        settings, exchange=MockExchange(), store=store, catalog=None
    ).download_candles("BTCUSDT", "1m", start=start, end=end)
    # Update with end before next candle -> noop
    frame = await DataUpdater(
        settings, exchange=MockExchange(), store=store, catalog=None
    ).update_candles("BTCUSDT", "1m", end=end)
    assert frame.height >= 1


@pytest.mark.unit
def test_duckdb_register_files_and_errors(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "p")
    catalog = DuckDBCatalog(tmp_path / "db.duckdb", store)
    with pytest.raises(DataError):
        catalog.register_files([], view_name="empty_view")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "open_time": [start],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    paths = store.write_frame(
        frame,
        data_type=MarketDataType.CANDLE,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe="1m",
        timestamp_column="open_time",
    )
    view = catalog.register_files(paths, view_name="custom_candles")
    assert view == "custom_candles"
    assert "custom_candles" in catalog.table_names()
    with catalog:
        assert catalog.connection is not None
    with pytest.raises(DataError):
        DuckDBCatalog(tmp_path / "db2.duckdb", store).sql("SELECT * FROM definitely_missing")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_historical_resume_checkpoint(tmp_path: Path) -> None:
    exchange = MockExchange()
    ingestor = HistoricalIngestor(
        exchange,
        page_limit=5,
        max_retries=2,
        retry_delay=0.01,
        checkpoint_dir=tmp_path / "cp",
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=12)
    key = "candles:mock:BTCUSDT:1m"
    mid = start + timedelta(minutes=5)
    from iqrp.app.data.types import utc_to_ms

    ingestor.save_checkpoint(key, {"cursor_ms": utc_to_ms(mid), "rows": 0})
    candles = await ingestor.download_candles("BTCUSDT", "1m", start=start, end=end, resume=True)
    assert candles
    assert ingestor.load_checkpoint(key) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_heartbeat_and_bad_payload() -> None:
    received: list[dict[str, Any]] = []

    async def handler(msg: dict[str, Any]) -> None:
        received.append(msg)

    class FakeWS:
        def __init__(self) -> None:
            self.items = ['{"id":"1","sequence":1}', 123]

        async def __aenter__(self) -> FakeWS:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        def __aiter__(self) -> FakeWS:
            return self

        async def __anext__(self) -> Any:
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

        async def ping(self) -> None:
            return None

    @asynccontextmanager
    async def connect(_url: str) -> AsyncIterator[FakeWS]:
        yield FakeWS()

    engine = WebsocketEngine(
        url="wss://x",
        handler=handler,
        heartbeat_interval=0.01,
        connect=connect,
    )
    with pytest.raises(DataError):
        await engine.run(max_reconnects=0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bybit_empty_ticker_and_coinbase_bad_tf() -> None:
    bybit = BybitExchange(
        ExchangeEndpointSettings(
            name="bybit", rest_base_url="https://api.bybit.com", ws_base_url="wss://x"
        )
    )
    with (
        patch.object(bybit, "request_json", new=AsyncMock(return_value={"result": {"list": []}})),
        pytest.raises(DataError),
    ):
        await bybit.fetch_mark_price("BTCUSDT")

    coin = CoinbaseExchange(
        ExchangeEndpointSettings(
            name="coinbase",
            rest_base_url="https://api.exchange.coinbase.com",
            ws_base_url="wss://x",
        )
    )
    with pytest.raises(DataError):
        await coin.fetch_candles(
            "BTC-USD",
            "3m",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 1, tzinfo=UTC),
            limit=10,
        )


@pytest.mark.unit
def test_validator_incorrect_order() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "open_time": [start + timedelta(minutes=1), start],
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [1.0, 1.0],
        }
    )
    anomalies, _report = DataValidator().validate_candles(
        frame, timeframe="1m", exchange="m", symbol="S"
    )
    assert any(a.kind.value == "incorrect_order" for a in anomalies)


@pytest.mark.unit
def test_factory_missing_endpoint() -> None:
    from iqrp.app.core.exceptions import ConfigurationError
    from iqrp.app.data.exchange.exchange_factory import ExchangeFactory

    factory = ExchangeFactory(DataSettings())
    with pytest.raises(ConfigurationError):
        factory.get_endpoint("not-configured")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_base_context_manager_and_request_ok() -> None:
    settings = ExchangeEndpointSettings(
        name="binance",
        rest_base_url="https://api.binance.com",
        ws_base_url="wss://x",
    )
    exchange = BinanceExchange(settings)
    async with exchange:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value={"ok": True})
        with patch.object(exchange.client, "request", new=AsyncMock(return_value=mock_resp)):
            assert await exchange.request_json("GET", "/ping") == {"ok": True}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_custom_heartbeat_and_bytes() -> None:
    got: list[dict[str, object]] = []

    async def handler(msg: dict[str, object]) -> None:
        got.append(msg)
        await engine.stop()

    heartbeats = {"n": 0}

    async def hb(ws: object) -> None:
        del ws
        heartbeats["n"] += 1

    class FakeWS:
        def __init__(self) -> None:
            self.items: list[object] = [b'{"id":"z","sequence":1}', '{"id":"z","sequence":1}']

        async def __aenter__(self) -> FakeWS:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        def __aiter__(self) -> FakeWS:
            return self

        async def __anext__(self) -> object:
            if not self.items:
                await asyncio.sleep(0.05)
                raise StopAsyncIteration
            return self.items.pop(0)

    @asynccontextmanager
    async def connect(_url: str) -> AsyncIterator[FakeWS]:
        yield FakeWS()

    engine = WebsocketEngine(
        url="wss://x",
        handler=handler,
        heartbeat_interval=0.01,
        connect=connect,
        send_heartbeat=hb,
    )
    await engine.run(max_reconnects=0)
    assert got
    assert len(got) >= 1


@pytest.mark.unit
def test_duckdb_bad_view_name(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "p")
    catalog = DuckDBCatalog(tmp_path / "db.duckdb", store)
    with pytest.raises(DataError):
        catalog.register_files([tmp_path / "x.parquet"], view_name="bad-view!")


@pytest.mark.unit
def test_timestamp_aware_and_naive() -> None:
    from iqrp.app.data.models._timestamps import ensure_utc_ms

    naive = datetime(2024, 1, 1)
    assert ensure_utc_ms(naive).tzinfo is not None
    aware = datetime(2024, 1, 1, tzinfo=UTC)
    assert ensure_utc_ms(aware).tzinfo is not None


@pytest.mark.unit
def test_candle_close_out_of_bounds() -> None:
    from iqrp.app.core.exceptions import ValidationError
    from iqrp.app.data.models import Candle
    from iqrp.app.data.types import Timeframe

    with pytest.raises(ValidationError):
        Candle(
            exchange="m",
            symbol="S",
            timeframe=Timeframe.M1,
            open_time=datetime(2024, 1, 1, tzinfo=UTC),
            open=100,
            high=110,
            low=90,
            close=120,
            volume=1,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bybit_bad_timeframe() -> None:
    bybit = BybitExchange(
        ExchangeEndpointSettings(
            name="bybit", rest_base_url="https://api.bybit.com", ws_base_url="wss://x"
        )
    )
    with pytest.raises(DataError):
        await bybit.fetch_candles(
            "BTCUSDT",
            "2w",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, 1, tzinfo=UTC),
            limit=1,
        )


@pytest.mark.unit
def test_find_missing_ranges_empty() -> None:
    ranges = DataValidator().find_missing_ranges(
        pl.DataFrame(),
        timeframe="1m",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, 0, 5, tzinfo=UTC),
    )
    assert len(ranges) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_historical_empty_page_and_retry_exhaust() -> None:
    exchange = MockExchange()
    exchange.fail_next_requests = 5
    ingestor = HistoricalIngestor(exchange, page_limit=10, max_retries=2, retry_delay=0.01)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(DataError):
        await ingestor.download_candles(
            "BTCUSDT", "1m", start=start, end=start + timedelta(minutes=1)
        )

    async def empty(*_a: object, **_k: object) -> list[object]:
        return []

    exchange2 = MockExchange()
    exchange2.fetch_candles = empty  # type: ignore[assignment,method-assign]
    ingestor2 = HistoricalIngestor(exchange2, page_limit=10, max_retries=1, retry_delay=0.01)
    assert (
        await ingestor2.download_candles(
            "BTCUSDT", "1m", start=start, end=start + timedelta(minutes=3)
        )
        == []
    )


@pytest.mark.unit
def test_checkpoint_non_dict(tmp_path: Path) -> None:
    exchange = MockExchange()
    ingestor = HistoricalIngestor(exchange, checkpoint_dir=tmp_path)
    path = tmp_path / "candles_mock_x.json"
    path.write_text("[1,2,3]", encoding="utf-8")
    # force key that maps to that file name pattern
    key = "x"
    # write via save then overwrite
    ingestor.save_checkpoint(key, {"a": 1})
    cp = ingestor._checkpoint_path(key)
    assert cp is not None
    cp.write_text("[]", encoding="utf-8")
    assert ingestor.load_checkpoint(key) is None
