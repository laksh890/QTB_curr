# POV

Percentage-of-Volume (participation) execution: trade a target fraction of expected market volume over a horizon, hard-capped by `max_participation`, with optional dynamic throttling.

**Class:** `iqrp.app.execution.algorithms.POVAlgorithm`  
**Registry key:** `pov`

Related: [ExecutionAlgorithms](ExecutionAlgorithms.md) · [VWAP](VWAP.md) · [TWAP](TWAP.md)

---

## Behavior

1. Clip to approved residual.  
2. Estimate expected market volume ≈ `ADV × (horizon / trading_day_seconds)`.  
3. Scale `target_participation` by urgency multiplier; clamp to `max_participation`.  
4. Expected trade = `min(approved, expected_market_vol × target)`.  
5. Distribute across buckets using `volume_curve` / `expected_volume_path` or uniform weights.  
6. If `dynamic=True`, throttle by liquidity, spread bps, and fill rate (never increase beyond parent).  
7. Enforce per-bucket hard participation; high urgency may push residual into later buckets within caps.

---

## Constructor

```python
from iqrp.app.execution.algorithms import POVAlgorithm

algo = POVAlgorithm(
    target_participation=0.10,
    max_participation=0.20,
    n_slices=10,
    horizon_seconds=300.0,
    dynamic=True,
    default_urgency="NORMAL",
)
```

| Parameter | Default | Role |
|-----------|---------|------|
| `target_participation` | `0.10` | Desired POV rate |
| `max_participation` | `0.20` | Hard ceiling (also settings `risk.max_participation`) |
| `n_slices` | `10` | Bucket count (urgency-adjusted) |
| `horizon_seconds` | `300` | Schedule length |
| `dynamic` | `True` | Liquidity / spread / fill-rate throttle |

### Urgency multipliers on target

| Urgency | Multiplier |
|---------|------------|
| LOW | 0.7 |
| NORMAL | 1.0 |
| HIGH | 1.25 |
| CRITICAL | 1.5 |

After scaling, `target = min(target, max_participation)`.

---

## Plan API

```python
slices = algo.plan(
    50_000,
    {
        "mid": 25.0,
        "spread": 0.02,
        "adv": 2e6,
        "side": "buy",
        "urgency": "HIGH",
        "target_participation": 0.08,
        "max_participation": 0.15,
        "horizon_seconds": 1200,
        "volume_curve": [1, 1.2, 1.5, 1.5, 1.2, 1],
        "liquidity": 0.9,
        "fill_rate": 0.85,
        "depth_score": 1.0,
    },
)
assert sum(s.quantity for s in slices) <= 50_000 + 1e-6
for s in slices:
    assert s.metadata["target_participation"] <= s.metadata["max_participation"] + 1e-12
```

---

## Dynamic throttling

When enabled:

- `liquidity` / `depth_score` in `[0.05, 2.0]` scales size  
- Wide spreads (vs ~5 bps) reduce pace  
- `fill_rate` in `[0.05, 1.5]` scales size  
- Aggregate still hard-clipped to approved residual

Throttling can leave residual unfilled if market capacity or caps bind — the algorithm does not invent extra size to “catch up” beyond `max_participation`.

---

## Engine usage

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
report = engine.execute(
    {"XYZ": 25_000},
    current={"XYZ": 0},
    algo="pov",
    urgency="NORMAL",
    market_context={
        "XYZ": {
            "mid": 12.5,
            "spread": 0.01,
            "adv": 5e5,
            "volatility": 0.03,
            "target_participation": 0.10,
            "max_participation": 0.18,
        }
    },
)
```

---

## When to use

- Explicit ADV participation mandates from risk / compliance  
- Illiquid names where TWAP equal sizing would spike impact  
- Coupling with engine `risk.max_participation` as a second hard gate

Prefer [VWAP](VWAP.md) when tracking a volume *shape* matters more than a constant rate; [ImplementationShortfall](ImplementationShortfall.md) when arrival risk aversion should front-load.
