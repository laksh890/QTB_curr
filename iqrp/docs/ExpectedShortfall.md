# Expected Shortfall (ES)

Expected Shortfall (Conditional Value-at-Risk) estimators under `iqrp.app.risk.tail`.

ES is the expected loss **conditional on exceeding VaR**. Values are positive loss numbers. Confidences snap to **90 / 95 / 99%**.

Related: [VaR.md](VaR.md) · [MonteCarloRisk.md](MonteCarloRisk.md)

---

## Relation to CVaR and CTE

| Name | Module | Notes |
|------|--------|-------|
| **CVaR** | `iqrp.app.risk.tail.cvar` | Same numeric definition as ES for continuous distributions here |
| **Expected Shortfall** | `iqrp.app.risk.tail.expected_shortfall.expected_shortfall` | Thin wrapper renaming CVaR methods to `name="expected_shortfall"` |
| **CTE** | `conditional_tail_expectation` | Mean loss below a quantile **or** an absolute return threshold |

In this codebase, **ES ≡ CVaR** for historical / parametric / Monte Carlo paths. CTE is the explicit conditional-tail helper (optionally threshold-based).

```python
from iqrp.app.risk.tail.cvar import historical_cvar
from iqrp.app.risk.tail.expected_shortfall import expected_shortfall, conditional_tail_expectation

es = expected_shortfall(returns, confidence=0.95, method="historical")
cvar = historical_cvar(returns, confidence=0.95)
assert abs(es.value - cvar.value) < 1e-12
```

---

## Methods

### Historical ES

```python
from iqrp.app.risk.tail.cvar import historical_cvar
from iqrp.app.risk.tail.expected_shortfall import expected_shortfall

rm = historical_cvar(returns, confidence=0.95, horizon=1)
rm = expected_shortfall(returns, confidence=0.95, method="historical")
# Mean of returns at/below the α-quantile; reported as positive loss; ×√H
```

### Parametric (Gaussian) ES

```python
from iqrp.app.risk.tail.cvar import parametric_cvar

rm = parametric_cvar(returns, confidence=0.99, horizon=1)
# ES = -μ + σ · φ(z_α) / α   (loss form), then horizon-scaled
```

### Monte Carlo ES

```python
from iqrp.app.risk.tail.cvar import monte_carlo_cvar

rm = monte_carlo_cvar(
    returns,
    confidence=0.95,
    horizon=5,
    n_simulations=5000,
    seed=42,
)
```

Simulates i.i.d. normal horizon aggregates, takes the left-tail mean beyond the VaR quantile.

---

## Conditional Tail Expectation (CTE)

```python
from iqrp.app.risk.tail.expected_shortfall import conditional_tail_expectation

# Default: condition on returns <= empirical (1-c) quantile
cte = conditional_tail_expectation(returns, confidence=0.95, horizon=1)

# Explicit absolute threshold (return units)
cte_thr = conditional_tail_expectation(returns, threshold=-0.02, horizon=1)
```

Use CTE when governance specifies a fixed loss barrier rather than a VaR-linked quantile.

---

## Engine usage

```python
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings

engine = RiskIntelligenceEngine(RiskSettings.default())

es = engine.expected_shortfall(returns, confidence=0.95)
cvar = engine.cvar(returns, method="parametric", confidence=0.99)
cvar_mc = engine.cvar(returns, method="monte_carlo")
```

Hydra (`configs/risk/default.yaml`):

```yaml
es:
  method: historical   # historical | parametric | monte_carlo
  confidence: 0.95
```

`engine.expected_shortfall` uses `settings.es.method` and `monte_carlo.n_simulations` / `seed` for the MC path.

---

## Governance notes

- Prefer ES over VaR for limit setting when subadditivity matters.  
- Pair ES with [stress testing](StressTesting.md) — parametric ES understates crises.  
- Report ES confidence and method in `RiskReport.tail_risk` via `calculate_risk()`.
