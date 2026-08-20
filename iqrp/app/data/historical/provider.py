"""Provider request/response types and HistoricalDataProvider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProviderCapabilities:
    provider_id: str
    supported_frequencies: tuple[str, ...]
    data_class: str  # DEVELOPMENT DATA | PRODUCTION / INSTITUTIONAL DATA
    rate_limits: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    license_status: str = "UNKNOWN"


@dataclass
class ProviderRequest:
    instrument: str
    start: datetime | str
    end: datetime | str
    frequency: str
    adjustment_policy: str = "unadjusted"  # unadjusted | adjusted
    original_symbol: str | None = None
    exchange_timezone: str = "Asia/Kolkata"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    frame: pd.DataFrame
    provider: str
    source: str
    retrieval_timestamp: str
    requested_range: tuple[str, str]
    actual_range: tuple[str | None, str | None]
    frequency: str
    timezone: str
    original_timezone: str
    exchange_timezone: str
    adjustment_policy: str
    original_symbol: str
    normalized_symbol: str
    currency: str = "INR"
    license_status: str = "UNKNOWN"
    data_class: str = "DEVELOPMENT DATA"
    rate_limit_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    availability_timestamp_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "source": self.source,
            "retrieval_timestamp": self.retrieval_timestamp,
            "requested_range": list(self.requested_range),
            "actual_range": list(self.actual_range),
            "frequency": self.frequency,
            "timezone": self.timezone,
            "original_timezone": self.original_timezone,
            "exchange_timezone": self.exchange_timezone,
            "adjustment_policy": self.adjustment_policy,
            "original_symbol": self.original_symbol,
            "normalized_symbol": self.normalized_symbol,
            "currency": self.currency,
            "license_status": self.license_status,
            "data_class": self.data_class,
            "rate_limit_info": dict(self.rate_limit_info),
            "warnings": list(self.warnings),
            "availability_timestamp_available": self.availability_timestamp_available,
            "row_count": int(len(self.frame)),
            "metadata": dict(self.metadata),
        }


class HistoricalDataProvider(ABC):
    """Pluggable historical market-data provider (not yfinance-specific)."""

    provider_id: str = ""

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        ...

    @abstractmethod
    def list_instruments(self) -> list[str]:
        ...

    @abstractmethod
    def available_history(self, instrument: str, frequency: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def supported_frequencies(self) -> tuple[str, ...]:
        ...

    @abstractmethod
    def rate_limits(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def download(self, request: ProviderRequest) -> ProviderResponse:
        ...


class ProviderError(RuntimeError):
    """Provider / network / auth failure."""


class RateLimitError(ProviderError):
    """Provider rate limit exceeded."""


class EmptyResponseError(ProviderError):
    """Provider returned no rows."""


__all__ = [
    "EmptyResponseError",
    "HistoricalDataProvider",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "RateLimitError",
]
