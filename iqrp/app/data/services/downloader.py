"""High-level historical downloader service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

from iqrp.app.config.settings import AppSettings
from iqrp.app.data.exchange.base import BaseExchange
from iqrp.app.data.exchange.exchange_factory import ExchangeFactory
from iqrp.app.data.ingestion.historical import HistoricalIngestor
from iqrp.app.data.ingestion.scheduler import IngestionScheduler
from iqrp.app.data.storage.duckdb_catalog import DuckDBCatalog
from iqrp.app.data.storage.parquet_store import ParquetStore
from iqrp.app.data.types import MarketDataType, Timeframe
from iqrp.app.data.validation.validator import DataValidator


class DataDownloader:
    """Orchestrate multi-symbol/timeframe historical downloads into Parquet + DuckDB."""

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
        checkpoint = (
            Path(settings.storage.cache_dir) / settings.data.ingestion.checkpoint_dirname
        )
        self.ingestor = HistoricalIngestor(
            self.exchange,
            page_limit=settings.data.ingestion.page_limit,
            max_retries=settings.data.ingestion.max_retries,
            retry_delay=settings.data.ingestion.retry_delay_seconds,
            retry_backoff=settings.data.ingestion.retry_backoff,
            checkpoint_dir=checkpoint,
        )
        self.scheduler = IngestionScheduler(concurrency=settings.data.ingestion.concurrency)

    async def download_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        *,
        start: datetime,
        end: datetime,
        resume: bool = True,
    ) -> pl.DataFrame:
        await self.exchange.open()
        try:
            candles = await self.ingestor.download_candles(
                symbol, timeframe, start=start, end=end, resume=resume
            )
            if not candles:
                return pl.DataFrame()
            frame = pl.DataFrame([c.to_row() for c in candles])
            written = self.store.write_frame(
                frame,
                data_type=MarketDataType.CANDLE,
                exchange=self.exchange.name,
                symbol=self.exchange.normalize_symbol(symbol),
                timeframe=str(timeframe),
                timestamp_column="open_time",
            )
            if self.catalog is not None and written:
                self.catalog.register_data_type(MarketDataType.CANDLE)
            anomalies, report = self.validator.validate_candles(
                frame,
                timeframe=str(timeframe),
                exchange=self.exchange.name,
                symbol=self.exchange.normalize_symbol(symbol),
            )
            logger.info(
                "download_complete symbol={} tf={} rows={} gaps={} coverage={:.2f}",
                symbol,
                timeframe,
                report.row_count,
                report.gap_count,
                report.coverage_pct,
            )
            if anomalies:
                logger.warning("download_anomalies count={}", len(anomalies))
            return frame
        finally:
            await self.exchange.close()

    async def download_many(
        self,
        *,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        start: datetime,
        end: datetime,
    ) -> dict[str, pl.DataFrame]:
        symbols = symbols or list(self.settings.data.ingestion.symbols)
        timeframes = timeframes or list(self.settings.data.ingestion.timeframes)
        results: dict[str, pl.DataFrame] = {}

        async def worker(symbol: str, timeframe: str, s: datetime, e: datetime) -> None:
            # Fresh exchange client per task for connection isolation.
            exchange = self.factory.create(self.exchange.name)
            downloader = DataDownloader(
                self.settings,
                exchange=exchange,
                store=self.store,
                catalog=self.catalog,
            )
            frame = await downloader.download_candles(symbol, timeframe, start=s, end=e)
            results[f"{symbol}:{timeframe}"] = frame

        await self.scheduler.run_window(
            symbols=symbols,
            timeframes=timeframes,
            start=start,
            end=end,
            worker=worker,
        )
        self.store.storage_stats()
        return results
