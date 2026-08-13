"""Pydantic V2 settings models for IQRP.

Hydra composes YAML overlays; these models validate and freeze the result
so the rest of the platform consumes typed, immutable configuration.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Environment(StrEnum):
    """Supported runtime environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LoggingSettings(BaseModel):
    """Logging subsystem configuration."""

    model_config = {"frozen": True}

    level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = False
    console_logs: bool = True
    file_logs: bool = True
    log_dir: Path = Path("logs")
    log_filename: str = "iqrp.log"
    rotation: str = "100 MB"
    retention: str = "30 days"
    compression: str = "gz"
    enqueue: bool = True
    diagnose: bool = False
    backtrace: bool = False


class StorageSettings(BaseModel):
    """Data storage configuration (DuckDB / Arrow / filesystem)."""

    model_config = {"frozen": True}

    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/iqrp.duckdb")
    parquet_dir: Path = Path("data/parquet")
    cache_dir: Path = Path("data/cache")


class ExchangeEndpointSettings(BaseModel):
    """Per-exchange HTTP/WebSocket endpoint configuration."""

    model_config = {"frozen": True}

    name: str
    rest_base_url: str
    ws_base_url: str
    rate_limit_per_second: float = Field(default=10.0, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)


class DataIngestionSettings(BaseModel):
    """Historical download and websocket ingestion knobs."""

    model_config = {"frozen": True}

    default_exchange: str = "binance"
    symbols: tuple[str, ...] = ("BTCUSDT",)
    timeframes: tuple[str, ...] = ("1m", "5m", "1h")
    max_retries: int = Field(default=5, ge=1)
    retry_delay_seconds: float = Field(default=0.5, gt=0)
    retry_backoff: float = Field(default=2.0, ge=1.0)
    page_limit: int = Field(default=1000, ge=1, le=5000)
    concurrency: int = Field(default=4, ge=1)
    checkpoint_dirname: str = ".checkpoints"
    ws_heartbeat_interval_seconds: float = Field(default=15.0, gt=0)
    ws_reconnect_max_delay_seconds: float = Field(default=60.0, gt=0)
    ws_reconnect_base_delay_seconds: float = Field(default=1.0, gt=0)
    parquet_compression: str = "zstd"
    auto_register_duckdb: bool = True


class DataSettings(BaseModel):
    """Market-data engineering configuration."""

    model_config = {"frozen": True}

    ingestion: DataIngestionSettings = Field(default_factory=DataIngestionSettings)
    exchanges: tuple[ExchangeEndpointSettings, ...] = Field(
        default_factory=lambda: (
            ExchangeEndpointSettings(
                name="binance",
                rest_base_url="https://api.binance.com",
                ws_base_url="wss://stream.binance.com:9443/ws",
                rate_limit_per_second=10.0,
            ),
            ExchangeEndpointSettings(
                name="bybit",
                rest_base_url="https://api.bybit.com",
                ws_base_url="wss://stream.bybit.com/v5/public/spot",
                rate_limit_per_second=10.0,
            ),
            ExchangeEndpointSettings(
                name="coinbase",
                rest_base_url="https://api.exchange.coinbase.com",
                ws_base_url="wss://ws-feed.exchange.coinbase.com",
                rate_limit_per_second=5.0,
            ),
        )
    )


class AppSettings(BaseModel):
    """Root application settings validated after Hydra composition."""

    model_config = {"frozen": True}

    name: str = "iqrp"
    version: str = "0.1.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    seed: int = Field(default=42, ge=0)
    timezone: str = "UTC"
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    data: DataSettings = Field(default_factory=DataSettings)

    @field_validator("environment", mode="before")
    @classmethod
    def _coerce_environment(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        return self.environment is Environment.TESTING

    @property
    def is_development(self) -> bool:
        return self.environment is Environment.DEVELOPMENT
