# Value at Risk (VaR)

Institutional VaR estimators under `iqrp.app.risk.tail.var`, exposed via `RiskIntelligenceEngine.var()`.

All VaR values are **positive loss numbers** in return units. Supported confidence levels: **90% / 95% / 99%** (other inputs snap to the nearest of these).

---

## Methods

| Method | Key | Description |
|--------|-----|-------------|
| Historical | `historical` | Empirical quantile of past returns |
| Parametric | `parametric` | Gaussian mean/vol fit |
| Monte Carlo | `monte_carlo` | i.i.d. normal paths calibrated to history |
| Filtered Historical Simulation | `fhs` | Causal EWMA vol standardization + re-scale |

Horizon scaling uses √H for historical / parametric / FHS; Monte Carlo sums H-day shocks.

---

## Quick start

```python
import numpy as np
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings
from iqrp.app.risk.tail.var import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    filtered_historical_var,
)

returns = np.random.default_rng(42).normal(0.0005, 0.01, size=1000)

# Direct estimators
h = historical_var(returns, confidence=0.95, horizon=1)
p = parametric_var(returns, confidence=0.99, horizon=5)
m = monte_carlo_var(returns, confidence=0.90, horizon=1, n_simulations=5000, seed=42)
f = filtered_historical_var(returns, confidence=0.95, horizon=1, lambda_=0.94)

print(h.value, h.method, h.confidence)

# Engine (respects configs/risk/default.yaml)
engine = RiskIntelligenceEngine(RiskSettings.default())
v95 = engine.var(returns, method="historical", confidence=0.95)
v99 = engine.var(returns, method="fhs", confidence=0.99)
v_mc = engine.var(returns, method="monte_carlo", confidence=0.90, horizon=10)
```

---

## Historical VaR

```python
from iqrp.app.risk.tail.var import historical_var

rm = historical_var(returns, confidence=0.95, horizon=1)
# value = max(-quantile(returns, 1-c), 0) * sqrt(horizon)
```

Non-parametric; sensitive to sample window and outliers. Prefer for short, dense return histories.

---

## Parametric VaR

```python
from iqrp.app.risk.tail.var import parametric_var

rm = parametric_var(returns, confidence=0.95, horizon=1)
# Fits μ, σ; VaR from Normal(μ, σ) left quantile, loss form
```

Fast and smooth; understates fat tails. Pair with ES / stress for governance.

For production σ, prefer Volatility Forecasting outputs and document scaling separately — see [RiskIntegration.md](RiskIntegration.md).

---

## Monte Carlo VaR

```python
from iqrp.app.risk.tail.var import monte_carlo_var

rm = monte_carlo_var(
    returns,
    confidence=0.99,
    horizon=5,
    n_simulations=10_000,
    seed=42,
)
```

Draws i.i.d. normals with sample μ/σ, aggregates over `horizon`, then takes the empirical loss quantile. Seeded for reproducibility (architectural rule 8).

For correlated / bootstrap / copula scenario VaR, use [MonteCarloRisk.md](MonteCarloRisk.md) `ScenarioEngine`.

---

## Filtered Historical Simulation (FHS)

```python
from iqrp.app.risk.tail.var import filtered_historical_var

rm = filtered_historical_var(returns, confidence=0.95, horizon=1, lambda_=0.94)
```

1. Build **causal** EWMA volatility (uses only past returns at each t).  
2. Standardize residuals `z_t = r_t / σ_t`.  
3. Re-scale the empirical z-quantile by the **latest** EWMA σ.  

No future information enters the vol filter (architectural rule 4).

---

## Confidences

```python
for c in (0.90, 0.95, 0.99):
    print(c, historical_var(returns, confidence=c).value)
```

Unsupported levels are snapped to the nearest of `{0.90, 0.95, 0.99}`.

---

## Engine configuration

```yaml
# iqrp/configs/risk/default.yaml
var:
  method: historical   # historical | parametric | monte_carlo | fhs
  confidence: 0.95
  horizon: 1
  n_simulations: 5000
```

```python
from iqrp.app.risk import RiskSettings

settings = RiskSettings.from_hydra(overrides=["var.method=fhs", "var.confidence=0.99"])
engine = RiskIntelligenceEngine(settings)
engine.var(returns)  # uses configured method / confidence / horizon
```

---

## Interpretation

- Report VaR alongside [Expected Shortfall](ExpectedShortfall.md) — VaR is not coherent; ES is.  
- Multi-asset books: aggregate returns with weights first, or use portfolio simulation in [MonteCarloRisk.md](MonteCarloRisk.md).  
- Live path: only point-in-time returns up to decision time; never include the bar being decided.
