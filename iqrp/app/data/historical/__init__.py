"""Historical equity/index market-data acquisition (research / development).

Pluggable providers normalize into the backtesting canonical OHLCV schema and
register with :class:`~iqrp.app.backtesting.data.dataset_registry.DatasetRegistry`.

Development / free provider data is **not** institutional-grade.
"""

from __future__ import annotations

from iqrp.app.data.historical.binance_vision import BinanceVisionHistoricalProvider
from iqrp.app.data.historical.cache import HistoricalCache
from iqrp.app.data.historical.pipeline import AcquisitionPipeline, AcquisitionResult
from iqrp.app.data.historical.provenance import DatasetProvenance, data_class_label
from iqrp.app.data.historical.provider import (
    HistoricalDataProvider,
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
)
from iqrp.app.data.historical.provider_registry import get_provider, list_providers, register_provider
from iqrp.app.data.historical.yahoo_finance import YahooFinanceHistoricalProvider

__all__ = [
    "AcquisitionPipeline",
    "AcquisitionResult",
    "BinanceVisionHistoricalProvider",
    "DatasetProvenance",
    "HistoricalCache",
    "HistoricalDataProvider",
    "ProviderCapabilities",
    "ProviderRequest",
    "ProviderResponse",
    "YahooFinanceHistoricalProvider",
    "data_class_label",
    "get_provider",
    "list_providers",
    "register_provider",
]
