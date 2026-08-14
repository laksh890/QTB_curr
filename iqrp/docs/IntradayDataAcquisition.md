# Intraday Historical Data Acquisition & Normalization

Data infrastructure only — **not** strategy alpha, **not** profitability claims.

## Workflow

```
acquire → raw → normalize → validate → canonical parquet → register → research/backtest
```

CLI:

```bash
python -m iqrp.app.data.acquire \
  --provider yahoo_finance \
  --instrument NIFTY50 \
  --start 2026-08-06 \
  --end 2026-08-14 \
  --frequency 1m \
  --output data/nifty50 \
  --adjustment-policy unadjusted \
  --derive 5m,15m,30m,1h

python -m iqrp.app.data.validate \
  --path data/nifty50/nifty50_intraday_1m.parquet \
  --frequency 1m
```

## Concepts

| Term | Meaning |
|------|---------|
| **DEVELOPMENT DATA** | Free / research provider feeds (e.g. Yahoo). Not institutional-grade. |
| **PRODUCTION / INSTITUTIONAL DATA** | Vendor feeds with contractual licensing (not claimed here). |
| **SOURCE frequency** | Native acquired bars (e.g. 1m). |
| **DERIVED frequency** | Session-aware OHLCV aggregation (e.g. 5m from 1m). Not equivalent to a native vendor 5m feed. |

## Provider abstraction

`HistoricalDataProvider` (`iqrp.app.data.historical.provider`) is pluggable. The framework does **not** depend solely on yfinance. Existing daily Yahoo NIFTY dataset remains valid.

## Timestamps

- Stored in **UTC**, timezone-aware.
- Naive provider timestamps require **explicit** `exchange_timezone` (no silent UTC assumption).
- Original + exchange timezones recorded in provenance.

## Sessions / calendar

Configurable `ExchangeCalendar` (NSE default 09:15–15:30 Asia/Kolkata). Weekends closed. Holidays / early closes / late opens must be configured — **not invented**.

Session assumptions: see `SESSION_BOUNDARY_ASSUMPTIONS` in `calendar.py`. Bars are never manufactured to fill boundaries.

## Validation

OHLC integrity, duplicates, ordering, zero/negative prices, negative volume, session expected vs actual bars, gap classification (`COMPLETE` / `MINOR_GAPS` / `MAJOR_GAPS` / `UNUSABLE`). **No silent repair.**

## Resampling

Deterministic, **session-bounded** aggregation. Provenance records `source_dataset_id`, source/derived frequency, aggregation method, checksum.

## Registry

Uses existing `DatasetRegistry`. Registration is **immutable** for a given `id@version` (new version required on change).

## Look-ahead / PIT

No future-filled OHLC, no price forward-fill, no future-based repair. If availability timestamps are absent, that limitation is recorded.

## Futures extensibility

Contract fields documented in provenance (`FUTURES_CONTRACT_FIELDS`). Continuous futures / rollovers are **not** fabricated here.

## Binance / BTCUSDT (crypto 24×7)

Public Binance Vision monthly kline archives (no API key):

```bash
python -m iqrp.app.data.acquire \
  --provider binance \
  --symbol BTCUSDT \
  --interval 1m \
  --start 2019-01-01 \
  --end 2026-08-01 \
  --output data/btcusdt \
  --derive 5m,15m,30m,1h
```

- `market_type=CRYPTO`, `timezone=UTC`, `continuous_market=true`, `session_model=24x7`
- Do **not** apply the NIFTY/NSE equity calendar to BTC
- `data_tier=DEVELOPMENT/RESEARCH`, `license_status=UNKNOWN`
- 1m = SOURCE; 5m/15m/30m/1h = DERIVED via existing `resample_session_aware`
- SOFTWARE VALIDATION ≠ STATISTICAL / ECONOMIC VALIDATION; no profitability claims
