# Exchange Adapters

## Design

Every venue implements `BaseExchange`. Selection is configuration-driven via
`ExchangeFactory` — adapters are registered by name and constructed from
`ExchangeEndpointSettings`.

```python
from iqrp.app.config import load_config, Environment
from iqrp.app.data import ExchangeFactory

settings = load_config(Environment.DEVELOPMENT)
exchange = ExchangeFactory(settings.data).create("binance")
```

## Supported adapters

| Name | Class | Notes |
|------|-------|-------|
| `binance` | `BinanceExchange` | Spot klines/trades/depth; futures funding/OI/mark/liquidations |
| `bybit` | `BybitExchange` | v5 spot + linear market endpoints |
| `coinbase` | `CoinbaseExchange` | Exchange API candles/trades/book; funding/OI N/A on spot |

## Extending

```python
from iqrp.app.data.exchange import register_exchange, BaseExchange

class MyVenue(BaseExchange):
    ...

register_exchange("myvenue", MyVenue)
```

Add a matching entry under `data.exchanges` in Hydra config (REST/WS URLs,
rate limit, timeout). No service code changes required.

## Responsibilities split

- **BaseExchange**: HTTP client lifecycle, rate limiting, error mapping
- **Adapter**: path construction, symbol normalization, payload parsing
- **Factory**: name → adapter, using configured endpoints only
