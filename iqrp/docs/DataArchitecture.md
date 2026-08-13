# Data Architecture

## Role

The `iqrp.app.data` package is the **single source of truth** for market data.
Every downstream module (regimes, forecasting, portfolio, risk, backtesting, live)
must read through the query API — never by scraping exchanges directly.

## Components

```text
Exchange adapters  →  Ingestion  →  Parquet partitions  →  DuckDB catalog
                           ↓
                     Validation / Repair
                           ↓
                     Query API (Polars)
```

| Package | Responsibility |
|---------|----------------|
| `exchange/` | `BaseExchange` + Binance / Bybit / Coinbase adapters + factory |
| `ingestion/` | Historical pagination, websocket engine, multi-job scheduler |
| `storage/` | ZSTD Parquet partitions + DuckDB view registration |
| `validation/` | Anomaly detection, quality reports, gap-only repair |
| `models/` | Pydantic contracts for all market-data products |
| `services/` | Downloader, updater, synchronizer, Polars query API |

## Data products

OHLCV, trades, order book snapshots, funding rates, open interest, mark price,
index price, liquidations.

All timestamps are timezone-aware **UTC** with millisecond precision at the
exchange boundary.

## Configuration

Hydra `data:` block in `iqrp/configs/config.yaml` controls:

- symbols / timeframes
- retry / pagination / concurrency
- parquet compression
- per-exchange REST/WS endpoints and rate limits

Nothing exchange-specific is hard-coded in services.

## Downstream contract

```python
from iqrp.app.config import load_config, Environment
from iqrp.app.data import get_candles

settings = load_config(Environment.DEVELOPMENT)
frame = get_candles(
    settings,
    exchange="binance",
    symbol="BTCUSDT",
    timeframe="1m",
)
```

Returns a **Polars** DataFrame.
