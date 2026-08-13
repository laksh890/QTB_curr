# Data Pipeline

Historical data ingestion for the operational Phase 13 backtest runner: local adapters → normalize → validate → register → chronological bar schedule → MARKET events.

> This package **does not download market data**. Users supply CSV / Parquet / Feather / Arrow files. Execution remains Phase 12; this pipeline feeds Phase 13 simulation.

---

## Purpose

Provide a single, auditable path from on-disk OHLCV into the event-driven runner with schema normalization, quality gates, dataset lineage (id / version / checksum), point-in-time helpers, universes, corporate-action loading, and continuous-futures utilities.

**Package:** `iqrp.app.backtesting.data`  
**Runner load path:** `iqrp.app.backtesting.runner.executor.load_market_frame`  
**Related:** [DataAdapters](DataAdapters.md) · [DatasetValidation](DatasetValidation.md) · [PointInTimeData](PointInTimeData.md) · [BacktestRunner](BacktestRunner.md) · [UserGuide](UserGuide.md)

---

## Architecture

```text
Local file (user-supplied)
        │
   ParquetAdapter / CSVAdapter / LocalFileProvider
        │
   normalize_frame (aliases, UTC timestamps, sort)
        │
   DatasetValidator → DataQualityReport
        │
   DatasetRegistry (id@version, checksum, coverage)
        │
   HistoricalDataset  (optional container / iteration)
        │
   load_market_frame → start/end/universe filters
        │
   bars_by_timestamp → MarketEvent payload{"bars": …}
        │
   EventPipeline.on_market → cascade
```

Optional helpers (import-only wrappers; do not replace platform modules):

- Corporate actions: `load_corporate_actions` / `corporate_actions_asof`
- Continuous contracts: `ContinuousContractBuilder` / `build_continuous_series`
- Universes: `UniverseSpec` / `resolve_universe` (PIT membership)

---

## Canonical schema

Required columns after alias normalization (`schema.REQUIRED_COLUMNS`):

| Column | Type | Notes |
|--------|------|-------|
| `timestamp` | tz-aware UTC | Aliases: `time`, `date`, `datetime`, … |
| `instrument` | str | Aliases: `symbol`, `ticker`, … |
| `open`, `high`, `low`, `close` | numeric | OHLC integrity checked |
| `volume` | numeric | Negative volume is critical |

Optional: `adj_close`, bid/ask sizes, `open_interest`, `settlement`, `vwap`, `contract`, `expiry`, `currency`, `exchange`. See [DataAdapters](DataAdapters.md) for alias map.

`normalize_frame` renames aliases, coerces UTC, sorts by `(timestamp, instrument)`.

---

## Dataset registry

`DatasetRegistry` persists JSON (`dataset_registry.json` by default):

```python
from iqrp.app.backtesting.data import DatasetRegistry

reg = DatasetRegistry("dataset_registry.json")
rec = reg.register_file(
    "/path/to/bars.parquet",
    dataset_id="my_universe",
    version="1.0.0",
    canonical_parquet=True,
)
assert reg.verify_checksum(rec.dataset_id, rec.version)
```

`DatasetRecord` fields: `dataset_id`, `version`, `source`, `path`, `checksum`, frequency, timezone, start/end, instrument_count, row_count, instruments, coverage_pct, corporate/liquidity flags, known_limitations, `registered_at`.

Key format: `{dataset_id}@{version}`.

---

## Runner integration

`load_market_frame(config)`:

1. Requires `config.dataset_path` (raises if missing / not found — **no download**).
2. Selects `CSVAdapter` or `ParquetAdapter` from `config.adapter`.
3. Loads + runs `DatasetValidator`; critical failures raise `ValueError`.
4. Applies optional `start` / `end` / `universe` filters.
5. Returns `(frame, detail)` where `detail` includes path, rows, `ok`, `critical_failures`, and full report dict.

Empty frames after filtering raise. Detail is stored on runner diagnostics as `data_detail`.

---

## Synthetic fixtures

Deterministic generators for tests and smoke demos only (not real markets, not investment advice):

```python
from pathlib import Path
from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv

write_synthetic_ohlcv(
    Path("fixtures/synthetic_bars.parquet"),
    n_days=60,
    instruments=["AAA", "BBB"],
    seed=7,
)
```

Also: `generate_synthetic_ohlcv`, `create_synthetic_ohlcv`, `HistoricalDataset.from_adapter`.

---

## Providers

| Type | Role |
|------|------|
| `DataProvider` | Abstract list / get_adapter / load / validate |
| `LocalFileProvider` | Index a directory of CSV/Parquet/Feather/Arrow |

Remote download providers are intentionally absent and must be introduced as audited subclasses if ever needed.

---

## Critical rules

| Rule | Detail |
|------|--------|
| User supplies data | Platform never fetches markets |
| UTC-aware timestamps | Naive timestamps fail validation |
| Critical → fail | No silent repair of schema/duplicates/invalid OHLC |
| Version + checksum | Register before production-style runs |
| PIT at bar boundary | Effective timestamps / universe as-of (see [PointInTimeData](PointInTimeData.md)) |

---

## Example end-to-end load

```python
from iqrp.app.backtesting.data import ParquetAdapter, DatasetValidator, HistoricalDataset

adapter = ParquetAdapter("fixtures/synthetic_bars.parquet", dataset_id="synthetic_demo")
ds = HistoricalDataset.from_adapter(adapter, validate=True, raise_on_critical=True)
for ts, cross_section in ds.iter_timestamps():
    # chronological cross-section per bar
    ...
```
