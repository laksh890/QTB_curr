# Validation & Repair

## Detection (`DataValidator`)

For OHLCV frames the validator reports:

| Check | Anomaly |
|-------|---------|
| Missing required columns | `missing_field` |
| Duplicate `open_time` | `duplicate_candle` |
| Non-ascending timestamps | `incorrect_order` |
| Step larger than timeframe | `timestamp_gap` / `missing_candle` |
| `volume < 0` | `negative_volume` |
| OHLC inconsistencies | `impossible_ohlc` |

## Quality report

`DataQualityReport` includes missing %, duplicates, gap count, coverage %,
oldest/newest record, and optional exchange latency.

## Repair (`DataRepair`)

Gap repair **downloads only missing ranges**:

1. Read local series
2. Compute missing inclusive ranges via `find_missing_ranges`
3. Fetch each gap through `HistoricalIngestor`
4. Append to Parquet and refresh DuckDB registration

Full history is never redownloaded for a partial hole.

## Orchestration

`DataSynchronizer.synchronize_candles()` runs repair + validation and returns
`(DataFrame, DataQualityReport)`.
