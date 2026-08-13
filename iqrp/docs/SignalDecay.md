# Signal Decay

Multi-horizon Information Coefficient analysis: how fast predictive power fades, the implied half-life, optimal research holding period, and live decay monitoring.

**Package:** `iqrp.app.alpha.research.decay` · `iqrp.app.alpha.monitoring.signal_decay`  
**Engine entry:** `AlphaResearchEngine.analyze_decay`  
**Parent:** [AlphaResearch](AlphaResearch.md) · Related: [SignalValidation](SignalValidation.md) · [SignalLifecycle](SignalLifecycle.md)

---

## Role in research

Decay diagnostics **inform holding periods** and retirement triggers. They do not approve alpha.

- Statistical significance alone ≠ alpha
- Historical Sharpe alone cannot approve
- Forward returns at horizon `h` are *labels*; the signal must already be point-in-time

Default horizons come from Hydra `research.horizons: [1, 2, 5, 10]`.

---

## Multi-horizon IC

```python
import numpy as np
from iqrp.app.alpha import AlphaResearchEngine
from iqrp.app.alpha.research.decay import analyze_decay, forward_returns

rng = np.random.default_rng(3)
returns = rng.normal(0, 0.01, 500)
signal = np.concatenate([[0.0], returns[:-1]])

eng = AlphaResearchEngine()
out = eng.analyze_decay(signal, returns, horizons=(1, 2, 5, 10, 20))
# or: analyze_decay(signal, returns, horizons=(1, 2, 5, 10))

print(out["ic"])          # {horizon: pearson IC}
print(out["rank_ic"])     # {horizon: rank IC}
print(out["hit_rate"])    # {horizon: directional hit rate}
print(out["half_life"])   # bars until |IC| ≈ ½ of first finite |IC|
print(out["optimal_hold"])  # horizon with max |IC|
```

`forward_returns(returns, horizon)` builds the label  
`r[t+1] + … + r[t+horizon]` with trailing NaNs where the future window is unavailable. Never feed that series back into signal construction.

---

## Half-life and optimal hold

| Field | Definition |
|-------|------------|
| `half_life` | Horizon where \|IC\| falls to half the first finite \|IC\| (linear interp on the grid; exponential extrapolation from the first two points if needed; else last horizon) |
| `optimal_hold` | Horizon on the evaluated grid with maximum \|IC\| |

Interpretation:

- Short half-life → high turnover → capacity and cost scrutiny ([SignalCapacity](SignalCapacity.md))
- Flat IC across horizons → possible slow factor; still validate with purged CV
- Sign flip across horizons → nonmonotonic relationship; document in `expected_relationship`

```python
from iqrp.app.alpha.research.decay import analyze_decay

decay = analyze_decay(signal, returns, horizons=(1, 2, 3, 5, 10, 21))
hold = decay["optimal_hold"]
hl = decay["half_life"]
# Align SignalDefinition.horizon with research optimal_hold when promoting
```

---

## Monitoring decay in production research

Live / paper monitoring uses rolling IC and decay alerts — still not trading approval.

```python
from iqrp.app.alpha.monitoring.signal_decay import rolling_ic, monitor_ic_decay
from iqrp.app.alpha.research.decay import forward_returns

fwd = forward_returns(returns, 1)
roll = rolling_ic(signal, fwd, window=60, step=5, rank=False)
# roll["ic"], roll["indices"] — rolling IC path

baseline = float(np.nanmean(roll["ic"][: max(1, len(roll["ic"]) // 4)]))
mon = monitor_ic_decay(roll, baseline_ic=baseline, collapse_ratio=0.3, warn_ratio=0.6)
# mon["status"] ∈ {HEALTHY, DECAYING, COLLAPSED}
```

Pair with `iqrp.app.alpha.monitoring.performance_decay` and `evaluate_retirement` when recent IC collapses relative to baseline ([SignalLifecycle](SignalLifecycle.md)).

---

## Research workflow

1. Compute decay grid on research sample (honest OOS preferred).
2. Set `SignalDefinition.horizon` near `optimal_hold`.
3. Ensure backtest turnover matches that horizon.
4. Re-estimate half-life on a schedule; material lengthening/shortening triggers review.
5. Do **not** approve solely because short-horizon IC is high.

```python
defn_horizon = int(out["optimal_hold"])
# Update definition before approve; re-run validate at that horizon
```

Disclaimer returned by `analyze_decay`:

> Decay diagnostics inform research holding periods. Statistical significance alone ≠ alpha. Historical Sharpe alone cannot approve.
