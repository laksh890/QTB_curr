# Signal Discovery

Candidate generation for institutional alpha research. Discovery **emits research candidates**, never approved alpha, and never claims profitability.

**Package:** `iqrp.app.alpha.discovery`  
**Orchestrator:** `CandidateGenerator` (also via `AlphaResearchEngine.discover`)  
**Parent:** [AlphaResearch](AlphaResearch.md)

---

## Core rule

**Statistical significance ≠ alpha.** A feature that screens at IC ≥ 0.02 is a *lead*. It still needs an economic hypothesis, leakage-safe validation, capacity analysis, and lifecycle approval before it is research-approved — and Risk Intelligence before it trades.

Discovery templates set `claims_profitability=False` and register (when enabled) at `SignalStatus.CANDIDATE`.

---

## Sources of candidates

| Source | API | Input | Output |
|--------|-----|-------|--------|
| Time-series templates | `from_time_series` / `build_time_series_candidates` | returns, optional volume/prices | momentum, mean-reversion, trend, vol, volume signals |
| Feature / statistical screen | `from_statistical_screen` / `screen_features` | `features` dict + target | IC-screened leads with scaffold hypothesis text |
| Symbolic formulas | `from_formulas` / `evaluate_expression` | series + op stacks | custom transform candidates |
| Forecasts | `from_forecasts` | forecast series + hypothesis | wrapped forecast signal |
| Cross-section | `from_cross_section` | feature panel | rank / z-score style CS candidates |
| Event-based | event discovery helpers | event mask + series | event-window candidates |
| Alternative data | alternative helpers | alt series | alt-data candidates |
| Unified | `discover_all` / `engine.discover` | any combination | bundled `DiscoveryResult` |

---

## Quick start

```python
import numpy as np
from iqrp.app.alpha import AlphaResearchEngine, AlphaSettings
from iqrp.app.alpha.discovery.candidate_generator import CandidateGenerator

rng = np.random.default_rng(0)
returns = rng.normal(0, 0.01, 400)
features = {
    "mom_20": np.concatenate([[0.0], np.cumsum(returns[:-1])]),
    "noise": rng.normal(size=400),
}
target = np.concatenate([returns[1:], [0.0]])  # 1-bar forward label

# Via engine
eng = AlphaResearchEngine(AlphaSettings.default())
candidates = eng.discover(
    returns=returns,
    features=features,
    forecasts=features["mom_20"],
    forecast_hypothesis=(
        "Forecast series encodes expected residual return from underreaction; "
        "not a profitability claim until validated."
    ),
)
assert all(c["claims_profitability"] is False for c in candidates)

# Via CandidateGenerator directly
gen = CandidateGenerator(owner="research", auto_register=True)
result = gen.from_statistical_screen(
    features,
    target,
    min_abs_ic=0.02,
    economic_hypothesis=(
        "Screened association is a statistical lead only; an economic "
        "mechanism must be articulated before promotion. "
        "Statistical significance alone ≠ alpha."
    ),
)
print(result.to_dict()["disclaimer"])
```

---

## Time-series templates

Configured by Hydra `discovery.*` lookbacks (`momentum_lookbacks`, `mean_reversion_lookbacks`, `trend_fast` / `trend_slow`, volatility/volume windows).

```python
from iqrp.app.alpha.discovery.candidate_generator import CandidateGenerator

gen = CandidateGenerator(auto_register=False)
ts = gen.from_time_series(
    returns,
    volume=np.abs(rng.normal(1e6, 1e5, 400)),
    prices=100 * np.exp(np.cumsum(returns)),
    momentum_lookbacks=(10, 20),
    mean_reversion_lookbacks=(5, 10),
)
# Each definition carries signal_type + economic_hypothesis scaffold
for d in ts.definitions:
    assert d.economic_hypothesis  # scaffold text; refine before APPROVED
    assert "candidate" in d.tags or d.signal_type in {
        "momentum", "mean_reversion", "trend", "volatility", "volume", "custom"
    }
```

Helpers are **point-in-time**: lookbacks use only past bars. Forward returns used as targets must not leak into signal construction.

---

## Feature / forecast discovery

### Statistical screen

```python
from iqrp.app.alpha.discovery.statistical import screen_features, candidates_to_signals

screens = screen_features(features, target, min_abs_ic=0.02, owner="research")
# screens are StatisticalCandidate objects — leads, not alpha
signals = candidates_to_signals(
    screens,
    features,
    economic_hypothesis=(
        "Association with forward returns motivates research; articulate "
        "an economic mechanism before promotion. "
        "Statistical significance alone ≠ alpha."
    ),
)
```

Defaults: `statistical_min_abs_ic=0.02`, `statistical_min_obs=30` in `configs/alpha/default.yaml`.

### Forecasts

```python
result = gen.from_forecasts(
    forecast=features["mom_20"],
    name="fi_residual_mu",
    economic_hypothesis=(
        "Forecast Intelligence residual μ reflects slow mean reversion of "
        "idiosyncratic mispricing after risk-factor adjustment."
    ),
    horizon=5,
)
```

Forecasts are **wrapped as candidates**. Forecast quality metrics do not auto-approve alpha.

---

## Symbolic and statistical formulas

```python
result = gen.from_formulas(
    series={"r": returns, "v": np.abs(returns)},
    formulas=[
        (
            "vol_scaled_mom",
            [("lag", {"x": "r", "k": 1}), ("div", {"a": "r", "b": "v"})],
            "Volatility-scaled lagged return proxies risk-adjusted underreaction; "
            "expected positive relationship at short horizons.",
        ),
    ],
)
```

Each formula tuple is `(name, ops, economic_hypothesis)`. Missing mechanism text is not acceptable for later APPROVED status.

---

## Cross-section, events, alternative

```python
# Cross-sectional rank of a (T, N) panel → candidate for asset_index
panel = rng.normal(size=(200, 50))
cs = gen.from_cross_section(panel, asset_index=0, method="rank")

# Unified discovery
all_cands = gen.discover_all(
    returns=returns,
    features=features,
    target=target,
    forecast=features["mom_20"],
    forecast_hypothesis="See FI residual underreaction hypothesis.",
    event_mask=(np.abs(returns) > 0.03),
)
```

---

## Registration behavior

| Setting | Behavior |
|---------|----------|
| `auto_register=true` (default) | Each candidate registered in `SignalRegistry` as `CANDIDATE` |
| `auto_register=false` | Signals/definitions returned only; caller registers explicitly |

```python
from iqrp.app.alpha import AlphaResearchEngine

eng = AlphaResearchEngine()
cands = eng.discover(returns=returns, features=features, auto_register=False)
for c in cands:
    if c["definition"] is not None:
        from iqrp.app.alpha import SignalDefinition
        eng.register(SignalDefinition.from_dict(c["definition"]), signal=c["signal"])
```

---

## What discovery does *not* do

- Does not set `SignalStatus.APPROVED`
- Does not skip multiple-testing accounting
- Does not imply capacity or net-of-cost profitability
- Does not bypass Risk Intelligence

Next steps: [SignalValidation](SignalValidation.md) → [BacktestValidation](BacktestValidation.md) → [SignalLifecycle](SignalLifecycle.md).
