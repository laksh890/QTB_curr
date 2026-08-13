# Storage

## Primary store — Apache Parquet

`ParquetStore` writes ZSTD-compressed Hive-style partitions:

```text
{parquet_dir}/{data_type}/exchange={ex}/symbol={sym}/timeframe={tf}/year=YYYY/month=MM/day=DD/data.parquet
```

Partition keys match the platform requirement:

`exchange / symbol / timeframe / year / month / day`

Writes are streaming per calendar day. Existing day files are merged and
de-duplicated on the timestamp column (keep last) to support resume and
incremental updates without full rewrites of the series.

## Catalog — DuckDB

`DuckDBCatalog` registers parquet globs as SQL views:

- `iqrp_candles`
- `iqrp_trades`
- `iqrp_orderbook`
- …

```python
catalog.register_all()
frame = catalog.sql("SELECT * FROM iqrp_candles WHERE symbol = 'BTCUSDT' LIMIT 10")
```

Views use `read_parquet(..., hive_partitioning=true, union_by_name=true)`.

## Paths

Configured under `storage:` and `data.ingestion.parquet_compression` — never
hard-coded in services.
