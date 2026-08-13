# Dataset Validation

Strict historical dataset quality gates for Phase 13 operational backtests. Critical failures fail the run; there is no silent repair.

---

## Purpose

`DatasetValidator` inspects schema, dtypes, timezone, ordering, duplicates, missing values, OHLC integrity, negative prices/volume, and coverage gaps, producing an institutional `DataQualityReport`.

**Module:** `iqrp.app.backtesting.data.dataset_validator`  
**Used by:** adapters (`validate`), `load_market_frame`, `HistoricalDataset.from_adapter`  
**Related:** [DataPipeline](DataPipeline.md) · [DataAdapters](DataAdapters.md) · [BacktestRunner](BacktestRunner.md)

---

## DataQualityReport

| Field | Meaning |
|-------|---------|
| `ok` | `True` iff `critical_failures` is empty |
| `critical_failures` | Human-readable critical messages |
| `issues` | List of `ValidationIssue` (`code`, `message`, `severity`, `count`, `details`) |
| `date_range`, `frequency`, `instrument_count`, `row_count` | Coverage summary |
| `missing_data`, `duplicate_data`, `invalid_data`, `coverage` | Structured stats |
| `timezone` | Expected `UTC` |
| `corporate_action_availability`, `liquidity_data_availability` | Flags |
| `known_limitations` | Warnings + metadata limitations |

`raise_if_critical()` raises `ValidationError` carrying the report.

Severities: `critical` | `warning` | `info`.

---

## Checks performed

| Code / area | Default severity | Behavior |
|-------------|------------------|----------|
| Missing required columns | critical | Fail immediately (pre-normalize) |
| Naive timestamps | critical (`fail_on_naive_timestamps`) | UTC-aware required |
| Non-UTC tz | warning | Allowed but flagged |
| Invalid/missing timestamps | critical | — |
| Non-numeric OHLCV | critical | — |
| Per-instrument timestamp order | critical (`require_sorted`) | — |
| Duplicate `(timestamp, instrument)` | critical (`fail_on_duplicates`) | — |
| Missing required values | critical (`fail_on_missing_required`) | — |
| High missing % (optional cols path) | warning | vs `max_missing_pct` (default 5%) |
| Intra-series gaps | warning | Daily threshold ≈ 4 days |
| Invalid OHLC (`high < low`, etc.) | critical (`fail_on_invalid_ohlc`) | — |
| Negative prices / volume | critical | Configurable |
| Non-positive prices | warning | — |

Constructor knobs allow softening individual fail flags for research sandboxes; production defaults are strict.

---

## API

```python
from iqrp.app.backtesting.data import DatasetValidator, ParquetAdapter

adapter = ParquetAdapter("/path/to/bars.parquet", dataset_id="user_ds")
report = adapter.validate(raise_on_critical=True)
assert report.ok
print(report.to_dict())
```

Standalone:

```python
from iqrp.app.backtesting.data import DatasetValidator, normalize_frame

validator = DatasetValidator()
report = validator.validate(frame, metadata={"dataset_id": "x", "version": "1.0.0"})
if not report.ok:
    raise SystemExit(report.critical_failures)
```

---

## Runner behavior

`load_market_frame` runs `DatasetValidator().validate(..., raise_on_critical=False)` then raises `ValueError` if `critical_failures` is non-empty. Preflight marks `dataset_validated` and fails the runner lifecycle to `FAILED` when the dataset is not OK.

Do not “fix” critical rows in place and continue; fix the source file or exclude the instrument/date range upstream, then re-register with a new version/checksum.

---

## Coverage

`frame_coverage` estimates observed vs expected bars (business-day approximation for `1d`). Registry `register_file` stores `coverage_pct` on the `DatasetRecord`.

---

## Critical rules

| Rule | Detail |
|------|--------|
| Critical → stop | Backtest must not proceed |
| No silent repair | Validator does not mutate bad OHLC into valid |
| UTC required | Naive timestamps are critical |
| Re-validate after edits | Checksum / version must change when data changes |
