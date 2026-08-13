# Synthetic Markets

What a `SimulatedMarket` contains and how to use it for institutional testing.

## Outputs

1. **Candles** — OHLCV plus bid/ask/spread, partitioned by symbol
2. **Trades** — synthetic prints with side/quantity
3. **Order book snapshots** — L2 depth ladders
4. **GroundTruth** — oracle regimes, volatility, drift, trend, transitions, events

## Microstructure

- Spread widens with volatility and liquidity stress
- Depth decays across levels; collapses under liquidity events
- Slippage via square-root impact (`SimulatedExecutionEngine`)

## Events

Injectable shocks (configurable probabilities):

- Flash crashes
- News shocks
- Gap opens
- Liquidity collapse
- Exchange outages
- Volatility spikes
- Momentum bursts
- Slow trends

## Asset classes

`stock`, `crypto`, `forex`, `commodity`, `index` — parameterization differs mainly
via drift/vol/liquidity defaults and session hours.

## Multi-asset

Set `n_assets > 1` and optional `correlation_matrix`. Shared regime path with
correlated innovations produces co-moving books suitable for portfolio / risk tests.

## Validation & charts

`SimulationValidator` checks mean, variance, distribution, autocorrelation, and
volatility against theoretical expectations (lenient bands when events/jumps are on).

SVG charts: price, returns, regimes, volatility, transition matrix, distribution,
autocorrelation.
