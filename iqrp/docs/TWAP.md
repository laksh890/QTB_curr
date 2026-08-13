# TWAP

Time-Weighted Average Price execution: slice an approved parent residual across equal (or lightly depth-weighted) time buckets over a horizon, with optional participation caps and timing jitter.

**Class:** `iqrp.app.execution.algorithms.TWAPAlgorithm`  
**Registry key:** `twap`  
**Base:** `ExecutionAlgorithm`

Related: [ExecutionAlgorithms](ExecutionAlgorithms.md) · [VWAP](VWAP.md) · [POV](POV.md)

---

## Behavior

1. Resolve approved quantity via `approved_quantity` (residual / max clips).  
2. Choose slice count from `n_slices` or `ceil(horizon / interval)`, then reduce count for higher urgency.  
3. Allocate equal weights (or `depth` vector if provided).  
4. Apply ADV participation cap over the horizon fraction of the trading day.  
5. Redistribute residual into uncapped capacity without exceeding parent.  
6. Schedule `not_before_offset` with optional jitter; attach urgency-scaled limit hints.

Never invents quantity beyond the approved residual.

---

## Constructor

```python
from iqrp.app.execution.algorithms import TWAPAlgorithm

algo = TWAPAlgorithm(
    n_slices=5,
    horizon_seconds=300.0,
    interval_seconds=None,      # if set, drives n_slices = ceil(horizon/interval)
    participation_cap=None,     # e.g. 0.10 of expected ADV over horizon
    jitter=0.0,                 # fraction of slice interval
    seed=42,                    # reproducible jitter
    default_urgency="NORMAL",
)
```

| Parameter | Default | Role |
|-----------|---------|------|
| `n_slices` | `5` | Base bucket count |
| `horizon_seconds` | `300` | Schedule length |
| `interval_seconds` | `None` | Fixed spacing alternative |
| `participation_cap` | `None` | Max ADV participation over horizon |
| `jitter` | `0.0` | Relative timing noise in `[0, 1]` |
| `seed` | `None` | RNG for jitter |

---

## Plan API

```python
slices = algo.plan(
    parent_qty=10_000,
    market_context={
        "mid": 100.0,
        "spread": 0.02,
        "adv": 5e6,
        "side": "buy",
        "urgency": "HIGH",
        "horizon_seconds": 600,
        "n_slices": 8,
        "participation_cap": 0.08,
        "trading_day_seconds": 23400,
        "depth": [1, 1, 1.2, 1, 0.8, 1, 1, 1],  # optional soft weights
        "jitter": 0.1,
    },
)
assert sum(s.quantity for s in slices) <= 10_000 + 1e-6
```

Context overrides constructor defaults for `n_slices`, `horizon_seconds`, `participation_cap`, and `jitter`.

---

## Urgency effects

Higher urgency → fewer, larger slices (`n_slices_for_urgency`) and more aggressive `limit_price_hint` (willing to cross more of the spread). Participation hard caps still bind.

---

## Engine usage

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
report = engine.execute(
    {"AAPL": 5000},
    current={"AAPL": 0},
    algo="twap",
    urgency="NORMAL",
    market_context={"AAPL": {"mid": 190, "spread": 0.02, "adv": 4e7, "volatility": 0.02}},
)
```

---

## When to use

- Neutral schedule when no reliable volume curve is available  
- Compliance-style even pacing over a fixed window  
- Baseline benchmark before VWAP/POV/IS specialization

Prefer [VWAP](VWAP.md) when an intraday volume profile is trusted; [POV](POV.md) when participation rate is the primary constraint; [ImplementationShortfall](ImplementationShortfall.md) when arrival-price risk dominates.
