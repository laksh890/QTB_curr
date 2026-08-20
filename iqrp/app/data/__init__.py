"""Institutional market-data engineering layer.

Downstream models must consume data exclusively through
:mod:`iqrp.app.data.services.query`.
"""

from iqrp.app.data.exchange import BaseExchange, ExchangeFactory
from iqrp.app.data.models import (
    Candle,
    DataQualityReport,
    FundingRate,
    IndexPrice,
    Liquidation,
    MarkPrice,
    OpenInterest,
    OrderBook,
    Trade,
)
from iqrp.app.data.services import (
    DataDownloader,
    DataSynchronizer,
    DataUpdater,
    MarketDataQueryService,
    get_candles,
    get_funding,
    get_open_interest,
    get_orderbook,
    get_trades,
)
from iqrp.app.data.storage import DuckDBCatalog, ParquetStore
from iqrp.app.data.types import ExchangeId, MarketDataType, Timeframe
from iqrp.app.data.validation import DataRepair, DataValidator

__all__ = [
    "BaseExchange",
    "Candle",
    "DataDownloader",
    "DataQualityReport",
    "DataRepair",
    "DataSynchronizer",
    "DataUpdater",
    "DataValidator",
    "DuckDBCatalog",
    "ExchangeFactory",
    "ExchangeId",
    "FundingRate",
    "IndexPrice",
    "Liquidation",
    "MarkPrice",
    "MarketDataQueryService",
    "MarketDataType",
    "OpenInterest",
    "OrderBook",
    "ParquetStore",
    "Timeframe",
    "Trade",
    "get_candles",
    "get_funding",
    "get_open_interest",
    "get_orderbook",
    "get_trades",
]
