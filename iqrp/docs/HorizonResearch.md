# Horizon Research (Trading Horizon & Short-Horizon Research Engine)

Research capability under `iqrp.app.backtesting.horizon`. **Not** a new phase.
Does **not** redesign portfolio, execution, risk, or data architecture.

## Concepts

| Term | Meaning |
|------|---------|
| **Data timeframe** | Bar frequency of the market series used for the simulation (e.g. `5m`, `1D`). |
| **Signal timeframe** | Frequency at which signals are evaluated. Must be ≥ data timeframe (same or coarser after resampling). |
| **Holding period** | How long a position is held after entry, in bars (`1bar`…`40bar`) or wall-clock time (`30m`). |

Example representable combo: `data=5m`, `signal=15m`, `holding=30m` (≈6 bars of 5m data after signal downsample).

These are **research variables**. The framework does **not** assume daily / 1h / 30m / … is “correct”.

## Unavailable data

Native frequency is inferred from the dataset. A requested data timeframe **finer** than native is marked **`UNAVAILABLE`**. The engine **never fabricates** intraday bars from daily data.

Example with NIFTY daily-only:

- `1m` … `4h` → UNAVAILABLE  
- `1D` → available  

## Horizon sweeps

`HorizonResearchEngine.sweep()` expands a configurable grid of:

- data timeframes (default: 1m, 5m, 15m, 30m, 1h, 4h, 1D)
- signal timeframes (default: same list, filtered so signal ≥ data)
- holding bars (default: 1, 2, 3, 5, 10, 20, 40)

Equal/coarser frequencies are obtained by OHLCV downsampling only.

## Costs

Every evaluated cell applies configurable:

- commission_bps, spread_bps, slippage_bps
- optional financing / impact per period

Reports **gross vs net** P&L, alpha, and Sharpe. When gross looks strong but net collapses, classification includes **`COST-INEFFICIENT`**.

## Turnover & capacity

Turnover: daily / weekly / monthly / annualized, plus turnover per unit net alpha and P&L per unit turnover.

Capacity: scenarios across capital levels using the existing `CapacityModel`. All capacity output is labeled **`ESTIMATED / MODEL-BASED`** — not claimed market capacity.

## Signal half-life

Forward returns over 1, 2, 3, 5, 10, 20 bars with mean/median, volatility, hit rate, IC, and a decay profile.

## Walk-forward / OOS

Each candidate can be split into train / validation / out-of-sample (fractional or dated). Horizon selection must **not** use full-period in-sample Sharpe alone.

## Multiple testing

Sweeps record number of configurations / strategies / horizons / parameter combinations. Optional hooks to deflated Sharpe and BH adjustments when available. Best observed ≠ automatically significant.

## Horizon Research Score

Configurable weighted score (see `ranking.py` docstring). Default components:

net Sharpe, expectancy, drawdown, stability, OOS, turnover, costs, trade count, capacity, statistical confidence.

**Highest return alone is not the ranking objective.**

## Selection: BEST ROBUST vs BEST IN-SAMPLE

`select_best_robust_horizon()` applies configurable gates (OOS expectancy/Sharpe, drawdown, costs, trade count, neighborhood stability).

These concepts are **explicitly different**.

## Neighborhood robustness

Performance is compared across neighboring timeframes (multiplicative window). Spikes isolated to one horizon are flagged **`FRAGILE`**.

## Long / short / flat

Research strategies may emit LONG, SHORT, FLAT and reverse. Operational runs use `LongShortMomentumStrategy` through the existing BacktestRunner cascade (risk/execution remain authoritative). Overtrading is **diagnosed**, not auto-suppressed; optional cooldown is strategy-defined.

## Classifications

`ROBUST` | `PROMISING` | `FRAGILE` | `COST-INEFFICIENT` | `INSUFFICIENT_DATA` | `OOS_FAILURE` | `UNAVAILABLE`

## Outputs

- Horizon research **report** (`engine.report()`)
- Machine-readable **matrix** (`engine.matrix()`)

All results are **research / simulated / modelled** — not live performance claims.

## Example

```python
from iqrp.app.backtesting.data.synthetic import generate_synthetic_ohlcv
from iqrp.app.backtesting.horizon import HorizonResearchConfig, HorizonResearchEngine

frame = generate_synthetic_ohlcv(n_days=120, freq="1d", seed=7, instruments=["DEMO"])
cfg = HorizonResearchConfig(
    data_timeframes=["1m", "5m", "1D"],
    holding_bars=[1, 5],
    instrument="DEMO",
)
eng = HorizonResearchEngine(frame, config=cfg)
eng.sweep()
report = eng.report()
```
