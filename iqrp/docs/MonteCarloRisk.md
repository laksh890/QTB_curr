# Monte Carlo Risk Engine

Scenario generation for tail risk, stress, and portfolio P&L distributions.

Modules:

- `iqrp.app.risk.simulation.monte_carlo` — parametric & correlated MC  
- `iqrp.app.risk.simulation.bootstrap` — i.i.d. & block bootstrap  
- `iqrp.app.risk.simulation.copula` — Gaussian copula + empirical margins  
- `iqrp.app.risk.simulation.scenario_engine.ScenarioEngine` — unified runner → VaR/ES  

Defaults: `configs/risk/default.yaml` → `monte_carlo` (`n_simulations`, `horizon`, `seed`, `block_size`).

---

## Parametric Monte Carlo

i.i.d. normal paths calibrated to historical mean/vol.

```python
from iqrp.app.risk.simulation import parametric_monte_carlo

sim = parametric_monte_carlo(returns, n_simulations=5000, horizon=10, seed=42)
# sim["paths"] shape (n_sim, horizon); sim["terminal"] = row sums
print(sim["mu"], sim["sigma"], sim["mean_terminal"])
```

---

## Historical bootstrap

Resample past returns with replacement (preserves empirical margin, destroys serial dependence).

```python
from iqrp.app.risk.simulation import historical_bootstrap

sim = historical_bootstrap(returns, n_simulations=5000, horizon=10, seed=42)
```

---

## Block bootstrap

Moving-block bootstrap preserves short-run dependence.

```python
from iqrp.app.risk.simulation import block_bootstrap

sim = block_bootstrap(
    returns,
    n_simulations=2000,
    horizon=20,
    block_size=5,  # from settings.monte_carlo.block_size
    seed=42,
)
```

---

## Regime-conditioned simulation

There is no hard-coded crisis calendar. Condition on **caller-supplied regime labels** from Regime Intelligence (import-only):

```python
import numpy as np
from iqrp.app.risk.simulation import parametric_monte_carlo, historical_bootstrap

# regimes: length-T array from RegimeModel.predict(frame) — PIT only
crisis_mask = regimes == crisis_id
crisis_returns = returns[crisis_mask]

if crisis_returns.size >= 30:
    sim = parametric_monte_carlo(crisis_returns, n_simulations=5000, horizon=5, seed=42)
else:
    # Fallback: bootstrap full history but document thin sample
    sim = historical_bootstrap(returns, n_simulations=5000, horizon=5, seed=42)

# Soft conditioning: mixture of regime-specific sims weighted by predict_proba[-1]
weights = probs[-1]  # (K,)
terminals = []
for k in range(len(weights)):
    rk = returns[regimes == k]
    if rk.size == 0:
        continue
    tk = parametric_monte_carlo(rk, n_simulations=2000, horizon=1, seed=42 + k)["terminal"]
    terminals.append((float(weights[k]), tk))
```

Always filter with regimes known at decision time — never future regime labels.

---

## Correlated Monte Carlo

Multivariate normal paths with PSD-clipped covariance.

```python
from iqrp.app.risk.simulation import correlated_monte_carlo

mean = np.mean(asset_returns, axis=0)       # (N,)
cov = np.cov(asset_returns, rowvar=False)   # (N, N)
sim = correlated_monte_carlo(mean, cov, n_simulations=5000, horizon=5, seed=42)
# paths: (n_sim, horizon, n_assets); terminal: (n_sim, n_assets)
portfolio_terminal = sim["terminal"] @ weights
```

---

## Copula simulation

Gaussian copula dependence + **empirical** marginal quantiles (past data only).

```python
from iqrp.app.risk.simulation import gaussian_copula_simulate

sim = gaussian_copula_simulate(asset_returns, n_simulations=5000, seed=42)
# or pass correlation=custom_corr
samples = sim["samples"]  # (n_sim, n_assets)
pnl = samples @ weights
```

---

## ScenarioEngine

```python
from iqrp.app.risk.simulation import ScenarioEngine

engine = ScenarioEngine(n_simulations=5000, horizon=5, seed=42, block_size=5)

for method in ("parametric", "bootstrap", "block_bootstrap", "correlated", "gaussian_copula"):
    out = engine.run(
        asset_returns if method in ("correlated", "gaussian_copula") else returns,
        method=method,
        weights=weights,
        confidence=0.95,
    )
    print(method, out["var"]["value"], out["expected_shortfall"]["value"])
```

Returns terminal mean/std plus VaR and ES `RiskMeasure` dicts under the chosen method.

---

## With RiskIntelligenceEngine

```python
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings

rie = RiskIntelligenceEngine(RiskSettings.default())
# Engine MC VaR/CVaR uses settings.var / settings.monte_carlo
rie.var(returns, method="monte_carlo", confidence=0.99)
rie.cvar(returns, method="monte_carlo")
```

For full scenario suites beyond i.i.d. parametric, call `ScenarioEngine` directly and attach results to monitoring / reports.

---

## Reproducibility & scale

- Always pass an explicit `seed` (default 42 in config).  
- Start with 1e3–1e4 sims for interactive work; raise toward 1e5–1e6 for regulatory-grade tails when compute allows.  
- Vectorized NumPy paths; keep horizon modest for pre-trade latency.
