"""Historical data ingestion package for institutional backtesting.

Provides local CSV / Parquet / Arrow adapters, dataset containers, validation,
PIT-aware universes, corporate-action loading, and continuous futures helpers.

This package does not download market data.
"""

from __future__ import annotations

from iqrp.app.backtesting.data.adapter import DataAdapter
from iqrp.app.backtesting.data.continuous_contract import (
    AdjustmentMethod,
    ContinuousContractBuilder,
    ContinuousContractConfig,
    ContractSeriesKind,
    ContractSpec,
    RollEvent,
    RollRule,
    build_continuous_series,
)
from iqrp.app.backtesting.data.corporate_actions import (
    corporate_actions_asof,
    load_corporate_actions,
    normalize_corporate_actions,
)
from iqrp.app.backtesting.data.csv_adapter import CSVAdapter
from iqrp.app.backtesting.data.dataset import HistoricalDataset, create_synthetic_ohlcv
from iqrp.app.backtesting.data.dataset_registry import (
    DatasetRecord,
    DatasetRegistry,
    compute_checksum,
)
from iqrp.app.backtesting.data.dataset_validator import (
    DataQualityReport,
    DatasetValidator,
    ValidationError,
    ValidationIssue,
)
from iqrp.app.backtesting.data.metadata import (
    CoverageInfo,
    DatasetMetadata,
    InstrumentMetadata,
    metadata_from_frame,
)
from iqrp.app.backtesting.data.parquet_adapter import (
    ParquetAdapter,
    file_sha256,
    parquet_canonical_sha256,
)
from iqrp.app.backtesting.data.point_in_time import (
    LookaheadViolation,
    assert_no_lookahead,
    ensure_effective_timestamps,
    filter_features_asof,
    filter_frame_asof_df,
    filter_signals_asof,
    filter_universe_membership_asof,
)
from iqrp.app.backtesting.data.provider import DataProvider, LocalFileProvider
from iqrp.app.backtesting.data.schema import (
    ALL_KNOWN_COLUMNS,
    COLUMN_ALIASES,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    infer_frequency,
    normalize_frame,
)
from iqrp.app.backtesting.data.universe import (
    UniverseKind,
    UniverseSpec,
    continuous_futures_universe,
    custom_universe,
    futures_universe,
    historical_universe,
    index_constituents,
    instrument_list,
    resolve_universe,
    single_instrument,
)

__all__ = [
    "ALL_KNOWN_COLUMNS",
    "COLUMN_ALIASES",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "AdjustmentMethod",
    "CSVAdapter",
    "ContinuousContractBuilder",
    "ContinuousContractConfig",
    "ContractSeriesKind",
    "ContractSpec",
    "CoverageInfo",
    "DataAdapter",
    "DataProvider",
    "DataQualityReport",
    "DatasetMetadata",
    "DatasetRecord",
    "DatasetRegistry",
    "DatasetValidator",
    "HistoricalDataset",
    "InstrumentMetadata",
    "LocalFileProvider",
    "LookaheadViolation",
    "ParquetAdapter",
    "RollEvent",
    "RollRule",
    "UniverseKind",
    "UniverseSpec",
    "ValidationError",
    "ValidationIssue",
    "assert_no_lookahead",
    "build_continuous_series",
    "compute_checksum",
    "continuous_futures_universe",
    "corporate_actions_asof",
    "create_synthetic_ohlcv",
    "custom_universe",
    "ensure_effective_timestamps",
    "file_sha256",
    "filter_features_asof",
    "filter_frame_asof_df",
    "filter_signals_asof",
    "filter_universe_membership_asof",
    "futures_universe",
    "historical_universe",
    "index_constituents",
    "infer_frequency",
    "instrument_list",
    "load_corporate_actions",
    "metadata_from_frame",
    "normalize_corporate_actions",
    "normalize_frame",
    "parquet_canonical_sha256",
    "resolve_universe",
    "single_instrument",
]
