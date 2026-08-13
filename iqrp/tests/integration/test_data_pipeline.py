"""Integration tests for BTCUSDT download → Parquet → DuckDB → validation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iqrp.app.config.settings import AppSettings, DataSettings, StorageSettings
from iqrp.app.data.services.downloader import DataDownloader
from iqrp.app.data.services.query import MarketDataQueryService, get_candles
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType
from iqrp.app.data.validation.validator import DataValidator
from iqrp.tests.unit.data.mock_exchange import MockExchange


@pytest.mark.integration
@pytest.mark.asyncio
async def test_btcusdt_download_parquet_duckdb_validator(tmp_path: Path) -> None:
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
    store = ParquetStore(settings.storage.parquet_dir, compression="zstd")
    catalog = DuckDBCatalog(settings.storage.duckdb_path, store)
    downloader = DataDownloader(settings, exchange=exchange, store=store, catalog=catalog)

    start = datetime(2024, 6, 1, tzinfo=UTC)
    end = start + timedelta(hours=2)  # 121 one-minute candles inclusive
    frame = await downloader.download_candles("BTCUSDT", "1m", start=start, end=end)
    assert frame.height == 121

    files = store.list_parquet_files(MarketDataType.CANDLE)
    assert files
    assert all(f.suffix == ".parquet" for f in files)

    views = catalog.register_all()
    assert any(v.startswith("iqrp_candles") for v in views)
    sql_frame = catalog.sql("SELECT symbol, count(*) AS n FROM iqrp_candles GROUP BY symbol")
    assert sql_frame.height >= 1

    anomalies, report = DataValidator().validate_candles(
        frame, timeframe="1m", exchange="mock", symbol="BTCUSDT"
    )
    assert report.coverage_pct == 100.0
    assert report.gap_count == 0
    assert report.oldest_record == start
    assert not any(a.kind.value == "missing_candle" for a in anomalies)

    queried = get_candles(
        settings,
        exchange="mock",
        symbol="BTCUSDT",
        timeframe="1m",
        start=start,
        end=end,
    )
    assert queried.height == 121

    service = MarketDataQueryService(settings, store=store, catalog=catalog)
    assert service.get_candles(exchange="mock", symbol="BTCUSDT", timeframe="1m").height == 121
    catalog.close()
