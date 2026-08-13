# Implementation Shortfall

Arrival-price oriented scheduler balancing expected market impact against timing (volatility) risk — an Almgren–Chriss-style discrete trajectory over the approved residual.

**Class:** `iqrp.app.execution.algorithms.ImplementationShortfallAlgorithm`  
**Registry keys:** `is`, `implementation_shortfall`

Related: [ExecutionAlgorithms](ExecutionAlgorithms.md) · [Slippage](Slippage.md) · [ExecutionCosts](ExecutionCosts.md)

---

## Behavior

1. Clip to approved residual.  
2. Map urgency → risk aversion κ (or use explicit `risk_aversion` / context).  
3. Build continuous AC inventory path  
   `x(t) ∝ sinh(κ_ac · (T − t)) / sinh(κ_ac · T)`  
   with `κ_ac = sqrt(κ · σ² / η)`, then difference into trades.  
4. Blend toward equal sizing when spreads are wide and urgency is low/normal.  
5. On adverse mid vs arrival (and urgency ≠ LOW), front-load further.  
6. Attach more aggressive early limit hints under HIGH/CRITICAL.  
7. Never increase total quantity beyond approved residual.

Higher urgency → larger κ → more front-loaded schedule (trade faster, accept more impact to cut timing risk).

---

## Constructor

```python
from iqrp.app.execution.algorithms import ImplementationShortfallAlgorithm

algo = ImplementationShortfallAlgorithm(
    n_slices=8,
    horizon_seconds=300.0,
    impact_coeff=0.1,        # permanent impact η
    temporary_impact=0.05,   # temporary impact ε
    risk_aversion=None,      # default from urgency table
    default_urgency="NORMAL",
)
```

| Parameter | Default | Role |
|-----------|---------|------|
| `n_slices` | `8` | Discretization |
| `horizon_seconds` | `300` | Trading window `T` |
| `impact_coeff` | `0.1` | Permanent impact coefficient |
| `temporary_impact` | `0.05` | Temporary impact coefficient |
| `risk_aversion` | `None` | Override κ |

### Default risk aversion by urgency

| Urgency | κ |
|---------|---|
| LOW | 0.35 |
| NORMAL | 1.0 |
| HIGH | 2.5 |
| CRITICAL | 6.0 |

---

## Plan API

```python
slices = algo.plan(
    12_000,
    {
        "mid": 100.0,
        "spread": 0.04,
        "volatility": 0.025,
        "adv": 3e6,
        "side": "buy",
        "urgency": "HIGH",
        "arrival_price": 99.8,
        "decision_price": 99.7,
        "impact_coeff": 0.12,
        "temporary_impact": 0.06,
        "horizon_seconds": 900,
        "n_slices": 10,
    },
)
# Early slices typically larger under HIGH urgency
qtys = [s.quantity for s in slices]
assert abs(sum(qtys) - 12_000) < 1e-6
assert qtys[0] >= qtys[-1]
```

Slice metadata includes `kappa`, `dt`, `expected_impact_px`, `arrival_price`, `decision_price`.

---

## Engine usage

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
report = engine.execute(
    {"AAPL": 8_000},
    current={"AAPL": 0},
    algo="implementation_shortfall",  # or algo="is"
    urgency="HIGH",
    market_context={
        "AAPL": {
            "mid": 190.0,
            "spread": 0.03,
            "adv": 5e7,
            "volatility": 0.02,
            "arrival_price": 189.9,
        }
    },
)
# Post-trade IS attribution lives in report.post_trade
```

---

## Relation to TCA

The algorithm *schedules* under an IS objective. Measured implementation shortfall after fills is computed by:

- `iqrp.app.execution.transaction_costs.post_trade_cost_analysis`
- `iqrp.app.execution.analytics.implementation_shortfall`
- Engine `ExecutionReport.post_trade` / `analytics`

See [ExecutionCosts](ExecutionCosts.md) for delay / trading / fee / opportunity attribution.

---

## When to use

- Arrival-price or decision-price mandates  
- High opportunity-cost environments (news, short alpha half-life)  
- Explicit trade-off between impact and timing risk

Use [TWAP](TWAP.md)/[VWAP](VWAP.md) for neutral pacing; [POV](POV.md) for hard participation; `adaptive` when live microstructure should reshape the path continuously.
