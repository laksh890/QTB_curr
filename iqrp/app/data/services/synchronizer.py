"""Synchronize + repair stored series against exchange history."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from loguru import logger

from iqrp.app.config.settings import AppSettings
from iqrp.app.data.exchange.base import BaseExchange
from iqrp.app.data.exchange.exchange_factory import ExchangeFactory
from iqrp.app.data.models import DataQualityReport
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType, Timeframe
from iqrp.app.data.validation.repair import DataRepair
from iqrp.app.data.validation.validator import DataValidator


class DataSynchronizer:
    """Validate local data and repair gaps against the exchange."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        exchange: BaseExchange | None = None,
        store: ParquetStore | None = None,
        catalog: DuckDBCatalog | None = None,
    ) -> None:
        self.settings = settings
        self.factory = ExchangeFactory(settings.data)
        self.exchange = exchange or self.factory.create()
        self.store = store or ParquetStore(
            settings.storage.parquet_dir,
            compression=settings.data.ingestion.parquet_compression,
        )
        self.catalog = catalog
        if catalog is None and settings.data.ingestion.auto_register_duckdb:
            self.catalog = DuckDBCatalog(settings.storage.duckdb_path, self.store)
        self.validator = DataValidator()
        self.repair = DataRepair(
            self.exchange,
            self.store,
            self.catalog,
            page_limit=settings.data.ingestion.page_limit,
            max_retries=settings.data.ingestion.max_retries,
            retry_delay=settings.data.ingestion.retry_delay_seconds,
            retry_backoff=settings.data.ingestion.retry_backoff,
        )

    async def synchronize_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start: datetime,
        end: datetime,
    ) -> tuple[pl.DataFrame, DataQualityReport]:
        await self.exchange.open()
        try:
            frame = await self.repair.repair_candles(
                symbol, timeframe, start=start, end=end
            )
            _, report = self.validator.validate_candles(
                frame,
                timeframe=str(timeframe),
                exchange=self.exchange.name,
                symbol=self.exchange.normalize_symbol(symbol),
            )
            logger.info(
                "sync_complete symbol={} tf={} coverage={:.2f} gaps={}",
                symbol,
                timeframe,
                report.coverage_pct,
                report.gap_count,
            )
            return frame, report
        finally:
            await self.exchange.close()

    def quality_report(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        exchange_latency_ms: float | None = None,
    ) -> DataQualityReport:
        frame = self.store.read(
            MarketDataType.CANDLE,
            exchange=self.exchange.name,
            symbol=self.exchange.normalize_symbol(symbol),
            timeframe=str(timeframe),
            timestamp_column="open_time",
        )
        _, report = self.validator.validate_candles(
            frame,
            timeframe=str(timeframe),
            exchange=self.exchange.name,
            symbol=self.exchange.normalize_symbol(symbol),
            exchange_latency_ms=exchange_latency_ms,
        )
        return report
