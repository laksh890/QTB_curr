# Execution Algorithms

Parent → child slice planners for institutional execution. Algorithms schedule **how** an approved residual is worked; they never invent positions or exceed `approved_quantity` / residual.

**Package:** `iqrp.app.execution.algorithms`  
**Registry:** `iqrp.app.execution.registry`  
**Base:** `ExecutionAlgorithm.plan(parent_qty, market_context) -> list[ChildSlice]`

Related: [TWAP](TWAP.md) · [VWAP](VWAP.md) · [POV](POV.md) · [ImplementationShortfall](ImplementationShortfall.md) · [ExecutionPlatform](ExecutionPlatform.md)

---

## Design rules

1. Total planned quantity ≤ approved residual (`approved_quantity` in base helpers).  
2. Urgency changes slice count, participation, and limit aggression — **never** hard risk or kill switches.  
3. Point-in-time context only (`mid`, `spread`, `adv`, `volume_curve`, …).  
4. Floating residual rounding is hard-clipped onto the last slice.

---

## `ChildSlice`

| Field | Meaning |
|-------|---------|
| `quantity` | Child size (also `.qty`) |
| `not_before_offset` | Seconds from schedule start |
| `limit_price_hint` | Urgency-scaled limit around mid ± half-spread |
| `urgency` | Per-slice urgency |
| `metadata` | Algo-specific diagnostics |

---

## Urgency

`Urgency`: `LOW` | `NORMAL` | `HIGH` | `CRITICAL`

| Effect | LOW | NORMAL | HIGH | CRITICAL |
|--------|-----|--------|------|----------|
| Slice-size factor | 0.75 | 1.0 | 1.35 | 1.75 |
| Slice count | more | base | fewer | fewest |
| Limit aggression (vs half-spread) | −0.35 | 0.0 | +0.55 | +1.0 (cross) |
| POV target multiplier | 0.7× | 1.0× | 1.25× | 1.5× |
| IS risk aversion (default) | 0.35 | 1.0 | 2.5 | 6.0 |

```python
from iqrp.app.execution.algorithms import coerce_urgency, Urgency

urg = coerce_urgency("high")  # Urgency.HIGH
```

---

## Registry

```python
from iqrp.app.execution.registry import (
    available_algorithms,
    get_algorithm,
    register_algorithm,
)

available_algorithms()
algo = get_algorithm("twap", n_slices=6, horizon_seconds=600)
slices = algo.plan(5_000, {"mid": 100, "spread": 0.02, "adv": 1e6, "urgency": "NORMAL"})
```

| Name keys | Class | Doc |
|-----------|-------|-----|
| `twap` | `TWAPAlgorithm` | [TWAP](TWAP.md) |
| `vwap` | `VWAPAlgorithm` | [VWAP](VWAP.md) |
| `pov` | `POVAlgorithm` | [POV](POV.md) |
| `is`, `implementation_shortfall` | `ImplementationShortfallAlgorithm` | [ImplementationShortfall](ImplementationShortfall.md) |
| `adaptive` | `AdaptiveAlgorithm` | below |
| `arrival`, `arrival_price` | `ArrivalPriceAlgorithm` | arrival tracking |
| `market` | `MarketAlgorithm` | single aggressive slice |
| `limit` | `LimitAlgorithm` | passive limit schedule |
| `liquidity_seeking` | `LiquiditySeekingAlgorithm` | depth-aware |
| `opportunistic` | `OpportunisticAlgorithm` | spread/imbalance opportunistic |

`ExecutionEngine.execute(..., algo="vwap")` resolves via `get_algorithm`.

---

## Overview of planners

### TWAP

Equal (or depth-weighted) time buckets over `horizon_seconds`, optional interval, participation cap, timing jitter. See [TWAP](TWAP.md).

### VWAP

Weights from historical / intraday volume curve; optional adaptive blend with live pace; participation cap. See [VWAP](VWAP.md).

### POV

Target participation of expected market volume, hard `max_participation`, dynamic throttle on liquidity/spread/fill rate. See [POV](POV.md).

### Implementation Shortfall

Almgren–Chriss-style inventory trajectory balancing permanent/temporary impact vs timing risk; front-loads under high urgency. See [ImplementationShortfall](ImplementationShortfall.md).

### Adaptive

Reshapes a base schedule from live feedback:

- Wide spreads → slower / more passive limits  
- Thin liquidity → smaller slices  
- Vol spikes → slow unless urgency `HIGH`/`CRITICAL`  
- Poor fill rate → more aggressive limits (**not** more quantity)  
- Progress / adverse drift → redistribute remaining residual within parent

```python
from iqrp.app.execution.algorithms import AdaptiveAlgorithm

algo = AdaptiveAlgorithm(n_slices=10, horizon_seconds=300, base_participation=0.10)
slices = algo.plan(
    8_000,
    {
        "mid": 50.0,
        "spread": 0.05,
        "volatility": 0.03,
        "adv": 2e6,
        "fill_rate": 0.6,
        "liquidity": 0.8,
        "urgency": "HIGH",
        "side": "buy",
    },
)
```

### Market / Limit / Liquidity-seeking / Opportunistic / Arrival

- **Market** — minimal slicing; aggressive completion of residual.  
- **Limit** — schedule with passive limit hints.  
- **Liquidity-seeking** — sizes to displayed depth / ADV pockets.  
- **Opportunistic** — waits for favorable spread/imbalance (still PIT).  
- **Arrival** — arrival-price oriented schedule + `arrival_slippage_bps` / `track_arrival_performance` helpers.

---

## Market context keys (common)

| Key | Used by |
|-----|---------|
| `mid` / `price` | All (limits, costs) |
| `spread` | Limit hints, throttles |
| `adv` / `average_daily_volume` | Participation caps |
| `volatility` | IS, adaptive, slippage |
| `urgency`, `side` | Aggression / sign |
| `residual`, `approved_quantity`, `max_quantity` | Hard residual clip |
| `horizon_seconds`, `n_slices` | Schedule shape |
| `participation_cap` / `max_participation` | Caps |
| `volume_curve`, `live_volume_pace` | VWAP / POV |
| `arrival_price`, `decision_price` | IS / analytics |
| `trading_day_seconds` | Horizon as day fraction (default 23400) |

---

## Engine integration

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
report = engine.execute(
    {"IBM": 2000},
    current={"IBM": 0},
    algo="adaptive",
    urgency="HIGH",
    market_context={"IBM": {"mid": 180, "spread": 0.03, "adv": 3e6, "volatility": 0.02}},
)
```

After `algorithm.plan`, the engine asserts `sum(slice.qty) ≤ parent.residual_qty` and clips each child before submit.
