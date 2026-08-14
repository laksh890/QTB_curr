# Alpha Research & Signal Discovery Engine

Research platform under `iqrp.app.backtesting.alpha_research`.

**Not** a new architecture phase. **Not** a profitability claim.

## Pipeline

```
MARKET DATA → FEATURE → FEATURE VALIDATION → SIGNAL → SIGNAL VALIDATION
  → POSITION → RISK/EXECUTION (existing cascade when wired)
  → ACCOUNTING → PERFORMANCE → STATISTICAL VALIDATION
```

## Feature registry

Every feature has `feature_id`, version, description, inputs, output schema, timeframe,
lookback, parameters, dependencies, availability, warmup.

Reference research features: returns, log_returns, volatility, ATR, RSI, MA, EMA, MACD,
momentum, rolling_zscore, volume_change, VWAP_distance, range, true_range.

## Signal registry

Signals **reference features** and emit continuous / binary / categorical values including
LONG (+1) / SHORT (−1) / FLAT (0).

Reference families: Momentum, Mean Reversion, Breakout, Volatility, Volume, Trend, Price Action.

## Causality & leakage

Features at T use only information ≤ T. Automated tests:

- reject future_* columns
- future price shift must change causal features
- normalization must not use future observations
- feature must not equal 1-bar future return

## Multi-timeframe

`TimeframeContext` records feature / signal / execution timeframes. Alignment uses
causal `merge_asof(..., direction="backward")`.

## Alpha Research Score

Configurable weighted score (see `ranking.py`). Raw return is not the sole criterion.

## Classifications

`ROBUST_ALPHA` | `PROMISING_ALPHA` | `FRAGILE_ALPHA` | `COST_INEFFICIENT` |
`OOS_FAILURE` | `INSUFFICIENT_DATA` | `SAMPLE_TOO_SHORT`

## Short samples

Development NIFTY windows (~6 sessions) are explicitly marked **SAMPLE TOO SHORT**.
Do not claim statistical significance from them.
