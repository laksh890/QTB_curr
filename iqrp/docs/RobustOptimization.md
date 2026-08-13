# Robust Optimization

Uncertainty sets, distributionally robust mean–variance, and parameter-uncertainty (estimation-error) portfolios that avoid extreme allocations under noisy \(\mu\) / \(\Sigma\).

Package: `iqrp.app.portfolio.optimization` / `iqrp.app.portfolio.robust`  
Entry point: `optimize_robust`  
Helpers: `optimize_distributional_robust`, `optimize_parameter_uncertainty`, `optimize_robust_mean_variance`, uncertainty set builders

Related: [MeanVariance](MeanVariance.md) · [PortfolioConstruction](PortfolioConstruction.md) · [Covariance](PortfolioConstruction.md)

---

## Modes

`optimize_robust(..., mode=...)`:

| Mode | Behavior |
|------|----------|
| `distributional` (default) | Worst-case \(\mu\) in an uncertainty set, then MV |
| `box` / `interval` | Box uncertainty on \(\mu\) |
| `ellipsoidal` / `ellipsoid` | Ellipsoid \(\{m:(m-\mu)^\top(\tau\Sigma)^{-1}(m-\mu)\le\rho^2\}\) |
| `parameter` / `se` / `estimation` | Estimation-error / SE-driven uncertainty |

Hard box/budget constraints remain hard. Robustness operates on the objective / effective \(\mu\), not by widening constraint limits.

---

## Uncertainty sets

```python
from iqrp.app.portfolio.robust.uncertainty_sets import (
    box_uncertainty_mu,
    ellipsoidal_uncertainty_mu,
)

box = box_uncertainty_mu(mu, relative=0.2, kappa=0.5, cov=cov)
# box["lower"], box["upper"], box["delta"]

ell = ellipsoidal_uncertainty_mu(mu, cov, rho=1.0, tau=1.0)
# worst-case portfolio return ≈ w'μ − ρ √(w'(τΣ)w)
```

Box deltas: `absolute_i` if provided, else `relative·|μ_i| + kappa·σ_i`.

---

## Avoiding extreme allocations

Robust modes shrink aggressive corner solutions that classical MV produces under estimation error:

1. Worst-case \(\mu\) tilts away from overconfident high-return names.
2. Parameter uncertainty widens effective risk around poorly estimated means.
3. Upstream `stabilize_mu` still winsorizes / clips before MV sub-solves.
4. Hard `max_weight` continues to bind — robustness is not a substitute for concentration limits.

---

## Examples

```python
import numpy as np
from iqrp.app.portfolio.optimization import optimize_robust

mu = np.array([0.12, 0.04, 0.02])
cov = np.diag([0.04, 0.09, 0.16])

rob = optimize_robust(
    mu=mu, cov=cov, mode="ellipsoidal",
    risk_aversion=1.5, long_only=True, max_weight=0.4,
    names=["a", "b", "c"],
)

param = optimize_robust(
    mu=mu, cov=cov, mode="parameter",
    long_only=True, max_weight=0.4,
)

box = optimize_robust(
    mu=mu, cov=cov, mode="box",
    relative=0.25, long_only=True, max_weight=0.35,
)
```

Via engine:

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
eng.optimize(mu=mu, cov=cov, method="robust", mode="distributional", names=["a", "b", "c"])
```

On infeasibility or solver failure, the engine applies configured fallback without relaxing hard constraints.
