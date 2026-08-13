# Data Adapters

Local historical market-data adapters for Phase 13 operational backtests.

---

## Purpose

Adapters load user-supplied files into normalized OHLCV frames, expose metadata/validation helpers, and never perform remote market downloads.

**Package:** `iqrp.app.backtesting.data`  
**Base:** `DataAdapter`  
**Implementations:** `ParquetAdapter`, `CSVAdapter`  
**Provider:** `LocalFileProvider`  
**Related:** [DataPipeline](DataPipeline.md) · [DatasetValidation](DatasetValidation.md)

---

## Base interface (`DataAdapter`)

```python
from iqrp.app.backtesting.data import DataAdapter  # ABC

adapter.load(refresh=False) -> DataFrame
adapter.validate(frame=None, raise_on_critical=False) -> DataQualityReport
adapter.metadata(refresh=False) -> DatasetMetadata
adapter.available_instruments() -> list[str]
adapter.available_dates() -> list[datetime]
adapter.load_range(start, end) -> DataFrame
adapter.load_instrument(instrument) -> DataFrame
adapter.load_universe(instruments) -> DataFrame
adapter.clear_cache()
```

Constructor knobs: `dataset_id`, `version`, `source`, `validator`, `normalize` (default True → `normalize_frame` on load).

---

## ParquetAdapter

Loads `.parquet` / `.pq`, directories (multi-file parquet), `.feather` / `.arrow`, or an in-memory `pyarrow.Table`.

```python
from iqrp.app.backtesting.data import ParquetAdapter, file_sha256, parquet_canonical_sha256

adapter = ParquetAdapter(
    "/path/to/bars.parquet",
    dataset_id="user_nifty",
    version="1.0.0",
)
frame = adapter.load()
digest = adapter.checksum(canonical=True)  # column-sorted parquet bytes
```

| Helper | Behavior |
|--------|----------|
| `file_sha256(path)` | SHA-256 of raw file bytes |
| `parquet_canonical_sha256(path)` | SHA-256 after column-sorted uncompressed rewrite |
| `adapter.checksum(canonical=False)` | File/table checksum; `canonical=True` for parquet files |

Missing path → `FileNotFoundError`.

---

## CSVAdapter

Loads a single `.csv` or concatenates `*.csv` under a directory.

```python
from iqrp.app.backtesting.data import CSVAdapter

adapter = CSVAdapter(
    "/path/to/bars.csv",
    dataset_id="user_csv",
    version="1.0.0",
    read_csv_kwargs={"parse_dates": ["timestamp"]},
)
```

---

## Column aliases

Source names (case-insensitive) map to canonical columns via `COLUMN_ALIASES`. Important mappings:

| Source | Canonical |
|--------|-----------|
| `time`, `date`, `datetime`, `dt`, `ts`, `open_time` | `timestamp` |
| `symbol`, `ticker`, `asset`, `secid` | `instrument` |
| `o`/`h`/`l`/`c`, `vol`/`v` | `open`/`high`/`low`/`close`/`volume` |
| `adjusted_close`, `adjclose` | `adj_close` |
| `bidsize`/`asksize`, `oi`, `settle`, `venue` | bid_size / ask_size / open_interest / settlement / exchange |

First-wins on collisions. After rename, missing required columns raise in `normalize_frame`.

---

## LocalFileProvider

Indexes a root directory for `*.csv`, `*.parquet`, `*.pq`, `*.feather`, `*.arrow` (optional recursive). Resolves `dataset_id` by stem or relative path.

```python
from iqrp.app.backtesting.data import LocalFileProvider

provider = LocalFileProvider("data/historical")
print(provider.list_datasets())
adapter = provider.get_adapter("my_bars")
```

Unsupported suffixes raise `ValueError`. Unknown ids raise `KeyError`.

---

## Metadata

`metadata_from_frame` / `adapter.metadata()` produce `DatasetMetadata`: id, version, source, path, frequency (`infer_frequency`), timezone (`UTC`), start/end, instrument list/count, row_count, checksum (when set), liquidity/corporate flags, known_limitations, per-instrument `InstrumentMetadata`.

---

## Critical rules

- Adapters fail fast on missing files; they do not fetch remote data.
- Prefer Parquet with `canonical_parquet=True` checksums for registry reproducibility.
- Always validate before runner preflight in production workflows ([DatasetValidation](DatasetValidation.md)).
- Timestamps must be UTC-aware after normalization.
