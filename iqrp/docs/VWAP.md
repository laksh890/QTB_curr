# VWAP

Volume-Weighted Average Price execution: allocate an approved residual across buckets proportional to a historical or live volume curve, subject to participation caps.

**Class:** `iqrp.app.execution.algorithms.VWAPAlgorithm`  
**Registry key:** `vwap`  
**Helper:** `normalize_volume_curve`

Related: [ExecutionAlgorithms](ExecutionAlgorithms.md) · [TWAP](TWAP.md) · [POV](POV.md)

---

## Behavior

1. Clip to approved residual.  
2. Load `volume_curve` / `intraday_volume_profile` (flat profile fallback ≡ TWAP weights).  
3. Normalize / resample the curve to the target slice count.  
4. Optionally blend with `live_volume_pace` / `realized_volume_curve` when `adaptive=True`.  
5. Apply ADV participation cap; reallocate shortfall into volume-heavy buckets without exceeding parent.  
6. Emit timed `ChildSlice` list with volume-weight metadata.

---

## Constructor

```python
from iqrp.app.execution.algorithms import VWAPAlgorithm
import numpy as np

algo = VWAPAlgorithm(
    n_slices=None,                 # derive from curve length if omitted
    horizon_seconds=300.0,
    participation_cap=0.15,
    volume_curve=np.array([0.05, 0.08, 0.12, 0.15, 0.20, 0.15, 0.12, 0.08, 0.05]),
    adaptive=True,
    default_urgency="NORMAL",
)
```

| Parameter | Default | Role |
|-----------|---------|------|
| `n_slices` | `None` | Force bucket count (resamples curve) |
| `horizon_seconds` | `300` | Schedule length |
| `participation_cap` | `0.15` | Max ADV participation over horizon |
| `volume_curve` | `None` | Historical U-shape / custom profile |
| `adaptive` | `True` | Blend live pace into weights |

---

## Volume curve normalization

```python
from iqrp.app.execution.algorithms.vwap import normalize_volume_curve

weights = normalize_volume_curve([1, 2, 3, 2, 1], n=8)  # resample + sum to 1
```

Empty or zero curves fall back to uniform weights.

---

## Plan API

```python
slices = algo.plan(
    20_000,
    {
        "mid": 50.0,
        "spread": 0.01,
        "adv": 1e7,
        "side": "sell",
        "urgency": "NORMAL",
        "volume_curve": [0.1, 0.15, 0.25, 0.25, 0.15, 0.1],
        "live_volume_pace": [0.12, 0.18, 0.22, 0.2, 0.16, 0.12],
        "adaptive_blend": 0.35,   # weight on live pace
        "participation_cap": 0.12,
        "horizon_seconds": 1800,
    },
)
for s in slices:
    print(s.quantity, s.not_before_offset, s.metadata["volume_weight"])
```

---

## Urgency effects

Higher urgency reduces slice count (larger buckets) and increases limit aggression. Participation caps remain hard; unmet residual under a binding cap is left unplanned rather than breaching ADV limits.

---

## Engine usage

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
curve = [0.08, 0.1, 0.14, 0.18, 0.18, 0.14, 0.1, 0.08]
report = engine.execute(
    {"MSFT": 15_000},
    current={"MSFT": 0},
    algo="vwap",
    market_context={
        "MSFT": {
            "mid": 420.0,
            "spread": 0.04,
            "adv": 2e7,
            "volatility": 0.018,
            "volume_curve": curve,
        }
    },
)
```

---

## When to use

- Liquid names with stable intraday volume profiles  
- Benchmarking against exchange VWAP  
- Adaptive pacing when live volume pace is observable

Use [TWAP](TWAP.md) when the curve is unreliable; [POV](POV.md) when a participation rate (not a shape) is mandated.
