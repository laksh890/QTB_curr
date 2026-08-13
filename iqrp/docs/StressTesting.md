# Stress Testing

Stress modules under `iqrp.app.risk.stress`, exposed via `RiskIntelligenceEngine.stress_test` and `reverse_stress`.

**Critical:** historical stress uses **caller-supplied** event windows/masks. The framework does **not** hard-code named crises (GFC, COVID, etc.).

---

## Historical stress

Replay portfolio (or asset) returns over an event mask or index window.

```python
import numpy as np
from iqrp.app.risk.stress import historical_stress

# Caller defines the crisis window — e.g. from a research calendar
event_indices = np.arange(420, 480)  # illustrative row indices in PIT history
# or boolean mask of length T
# or (start, end) pair

out = historical_stress(
    asset_returns,                 # (T,) or (T, N)
    event_window=event_indices,
    weights=weights,               # required for (T, N)
)
print(out["cumulative_return"], out["loss"], out["worst_drawdown"], out["n_event_days"])
```

```python
out = historical_stress(
    returns,
    event_mask=(dates >= start) & (dates <= end),  # caller-built
)
```

Engine:

```python
engine.stress_test(weights, returns=returns, event_indices=event_indices)
# → {"historical": {...}}
```

---

## Hypothetical stress

Apply additive return shocks: PnL = w · shock. Covariance provides diagnostic portfolio vol context.

```python
from iqrp.app.risk.stress import hypothetical_stress, ScenarioSpec

# Vector shocks
hyp = hypothetical_stress(weights, cov, shocks=np.array([-0.05, -0.08, -0.02]))

# Named scenario
spec = ScenarioSpec(
    name="rates_up_equity_down",
    shocks={"SPX": -0.10, "TLT": -0.03},
    description="Parallel risk-off sketch",
)
hyp = hypothetical_stress(weights, cov, spec, names=["SPX", "TLT", "FX"])
print(hyp["pnl"], hyp["loss"], hyp["portfolio_volatility"])
```

Engine:

```python
engine.stress_test(weights, shocks={"a": -0.1, "b": -0.05}, cov=cov)
engine.stress_test(weights, shocks=[-0.08, -0.08, -0.08])  # identity cov fallback
```

Extend with volatility / correlation / liquidity / gap / regime-transition shocks by constructing the shock vector or cov from upstream models, then calling `hypothetical_stress` — Risk does not invent those narratives.

---

## Reverse stress

Find the smallest shock **magnitude** along a direction that breaches a loss limit.

```python
from iqrp.app.risk.stress import reverse_stress

rev = reverse_stress(
    weights,
    direction=np.ones(len(weights)),  # normalized internally
    loss_limit=0.03,                  # e.g. max_daily_loss
)
print(rev["breach_possible"], rev["magnitude"], rev["signed_magnitude"])
```

Engine defaults `loss_limit` to `settings.limits.max_daily_loss`:

```python
engine.reverse_stress(weights, loss_limit=0.05, direction=np.array([1.0, -0.5, 0.2]))
```

Use reverse stress against drawdown, margin, liquidity, or concentration barriers by setting `loss_limit` (or mapping those constraints to an equivalent loss threshold) and choosing an economically meaningful `direction`.

---

## Scenario specifications

```python
from iqrp.app.risk.stress.scenarios import ScenarioSpec, apply_shock

spec = ScenarioSpec(name="gap_open", shocks=[-0.07, -0.04, -0.02])
apply_shock(weights, spec)
```

---

## Governance practice

1. Maintain an **external** event catalog (research DB / YAML) — never bake dates into `iqrp.app.risk`.  
2. Run historical + hypothetical + reverse as a package before limit changes.  
3. Attach results into `RiskReport` metadata / monitoring dashboards.  
4. Validate estimators on Simulation Engine paths ([SimulationEngine.md](SimulationEngine.md)) under high-vol, gap, and correlation-spike scenarios.
