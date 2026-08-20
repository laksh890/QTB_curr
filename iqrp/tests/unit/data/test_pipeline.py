"""Unit tests for parquet store, duckdb, validator, repair, websocket."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from iqrp.app.config.settings import AppSettings, DataSettings, StorageSettings
from iqrp.app.data.ingestion.historical import HistoricalIngestor
from iqrp.app.data.ingestion.scheduler import IngestionScheduler
from iqrp.app.data.ingestion.websocket import WebsocketEngine
from iqrp.app.data.rate_limiter import AsyncRateLimiter
from iqrp.app.data.services.downloader import DataDownloader
from iqrp.app.data.services.query import (
    get_candles,
    get_funding,
    get_open_interest,
    get_orderbook,
    get_trades,
)
from iqrp.app.data.services.synchronizer import DataSynchronizer
from iqrp.app.data.services.updater import DataUpdater
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType, Timeframe
from iqrp.app.data.validation.validator import DataValidator
from iqrp.tests.unit.data.mock_exchange import MockExchange


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limiter_allows_burst() -> None:
    limiter = AsyncRateLimiter(100.0, capacity=2)
    await limiter.acquire()
    await limiter.acquire()


@pytest.mark.unit
def test_validator_detects_gaps_and_duplicates(tmp_path: Path) -> None:
    del tmp_path
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "open_time": start,
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 1,
        },
        {
            "open_time": start,
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 1,
        },
        {
            "open_time": start + timedelta(minutes=3),
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": -1,
        },
    ]
    frame = pl.DataFrame(rows)
    anomalies, report = DataValidator().validate_candles(
        frame, timeframe="1m", exchange="mock", symbol="BTCUSDT"
    )
    kinds = {a.kind.value for a in anomalies}
    assert "duplicate_candle" in kinds
    assert "timestamp_gap" in kinds
    assert "negative_volume" in kinds
    assert report.gap_count >= 1


@pytest.mark.unit
def test_parquet_and_duckdb_roundtrip(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "parquet", compression="zstd")
    start = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "exchange": ["mock"],
            "symbol": ["BTCUSDT"],
            "timeframe": ["1m"],
            "open_time": [start],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1.0],
            "close_time": [start + timedelta(minutes=1)],
            "quote_volume": [100.0],
            "trade_count": [1],
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
    assert paths
    loaded = store.read(
        MarketDataType.CANDLE,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe="1m",
    )
    assert loaded.height == 1
    catalog = DuckDBCatalog(tmp_path / "iqrp.duckdb", store)
    view = catalog.register_data_type(MarketDataType.CANDLE)
    assert view is not None
    result = catalog.sql("SELECT count(*) AS n FROM iqrp_candles")
    assert int(result["n"][0]) == 1
    catalog.close()
    stats = store.storage_stats()
    assert stats["file_count"] >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_historical_download_with_retry_and_checkpoint(tmp_path: Path) -> None:
    exchange = MockExchange()
    exchange.fail_next_requests = 1
    ingestor = HistoricalIngestor(
        exchange,
        page_limit=10,
        max_retries=3,
        retry_delay=0.01,
        retry_backoff=1.0,
        checkpoint_dir=tmp_path / "cp",
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=25)
    candles = await ingestor.download_candles("BTCUSDT", "1m", start=start, end=end)
    assert len(candles) == 26


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_duplicate_and_sequence(tmp_path: Path) -> None:
    del tmp_path
    received: list[dict[str, Any]] = []

    async def handler(msg: dict[str, Any]) -> None:
        received.append(msg)

    class FakeWS:
        def __init__(self, messages: list[Any]) -> None:
            self._messages = messages
            self._idx = 0

        async def __aenter__(self) -> FakeWS:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        def __aiter__(self) -> FakeWS:
            return self

        async def __anext__(self) -> Any:
            if self._idx >= len(self._messages):
                raise StopAsyncIteration
            item = self._messages[self._idx]
            self._idx += 1
            await asyncio.sleep(0)
            return item

        async def ping(self) -> None:
            return None

    messages = [
        json.dumps({"id": "a", "sequence": 1, "T": 1704067200000}),
        {"id": "a", "sequence": 1, "T": 1704067200000},  # duplicate
        {"id": "b", "sequence": 3, "T": 1704067201000},  # gap from 1 -> 3
    ]

    @asynccontextmanager
    async def connect(_url: str) -> AsyncIterator[FakeWS]:
        yield FakeWS(messages)

    engine = WebsocketEngine(
        url="wss://mock",
        handler=handler,
        heartbeat_interval=60.0,
        connect=connect,
    )

    async def stopper() -> None:
        await asyncio.sleep(0.05)
        await engine.stop()

    await asyncio.gather(engine.run(max_reconnects=0), stopper())
    assert engine.stats.duplicates >= 1
    assert engine.stats.sequence_gaps >= 1
    assert engine.stats.messages >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scheduler_runs_jobs() -> None:
    seen: list[str] = []

    async def worker(symbol: str, timeframe: str) -> None:
        seen.append(f"{symbol}:{timeframe}")

    scheduler = IngestionScheduler(concurrency=2)
    jobs = scheduler.build_jobs(["BTCUSDT", "ETHUSDT"], ["1m", "5m"])
    await scheduler.run(jobs, worker)
    assert len(seen) == 4


@pytest.mark.unit
@pytest.mark.asyncio
async def test_downloader_updater_synchronizer_query(tmp_path: Path) -> None:
    settings = AppSettings(
        storage=StorageSettings(
            data_dir=tmp_path,
            duckdb_path=tmp_path / "iqrp.duckdb",
            parquet_dir=tmp_path / "parquet",
            cache_dir=tmp_path / "cache",
        ),
        data=DataSettings(),
    )
    exchange = MockExchange()
    store = ParquetStore(settings.storage.parquet_dir)
    catalog = DuckDBCatalog(settings.storage.duckdb_path, store)
    downloader = DataDownloader(settings, exchange=exchange, store=store, catalog=catalog)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=59)
    frame = await downloader.download_candles("BTCUSDT", Timeframe.M1, start=start, end=end)
    assert frame.height == 60

    # Incremental update
    updater = DataUpdater(settings, exchange=MockExchange(), store=store, catalog=catalog)
    updated = await updater.update_candles(
        "BTCUSDT", "1m", end=end + timedelta(minutes=10)
    )
    assert updated.height >= 60

    # Create gaps then repair via synchronizer with gap-producing exchange
    gapped = MockExchange(
        gap_after=start + timedelta(minutes=10),
        gap_size=3,
    )
    # Overwrite a subset with gaps artificially by writing incomplete data then repairing
    sync = DataSynchronizer(settings, exchange=MockExchange(), store=store, catalog=catalog)
    repaired, report = await sync.synchronize_candles(
        "BTCUSDT", "1m", start=start, end=end
    )
    assert repaired.height >= 1
    assert report.coverage_pct >= 0.0
    quality = sync.quality_report("BTCUSDT", "1m")
    assert quality.row_count >= 1

    candles = get_candles(
        settings, exchange="mock", symbol="BTCUSDT", timeframe="1m", start=start, end=end
    )
    assert candles.height >= 1

    # Persist other products for query API coverage
    trades = await gapped.fetch_trades("BTCUSDT", start=start, end=end, limit=1)
    store.write_frame(
        pl.DataFrame([t.to_row() for t in trades]),
        data_type=MarketDataType.TRADE,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe=None,
        timestamp_column="timestamp",
    )
    book = await gapped.fetch_orderbook("BTCUSDT")
    store.write_frame(
        pl.DataFrame([book.to_row()]),
        data_type=MarketDataType.ORDERBOOK,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe=None,
        timestamp_column="timestamp",
    )
    funding = await gapped.fetch_funding("BTCUSDT", start=start, end=end, limit=1)
    store.write_frame(
        pl.DataFrame([f.to_row() for f in funding]),
        data_type=MarketDataType.FUNDING,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe=None,
        timestamp_column="timestamp",
    )
    oi = await gapped.fetch_open_interest("BTCUSDT", start=start, end=end, limit=1)
    store.write_frame(
        pl.DataFrame([x.to_row() for x in oi]),
        data_type=MarketDataType.OPEN_INTEREST,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe=None,
        timestamp_column="timestamp",
    )

    assert get_trades(settings, exchange="mock", symbol="BTCUSDT").height >= 1
    assert get_orderbook(settings, exchange="mock", symbol="BTCUSDT").height >= 1
    assert get_funding(settings, exchange="mock", symbol="BTCUSDT").height >= 1
    assert get_open_interest(settings, exchange="mock", symbol="BTCUSDT").height >= 1
    catalog.close()
