# Mean–Variance Optimization

Mean–variance, global minimum variance, and maximum Sharpe (tangency) portfolios under hard box and budget constraints.

Package: `iqrp.app.portfolio.optimization`  
Entry points: `optimize_mean_variance`, `optimize_minimum_variance`, `optimize_maximum_sharpe`  
Facade: `PortfolioConstructionEngine.optimize(..., method=...)`

Related: [PortfolioConstruction](PortfolioConstruction.md) · [BlackLitterman](BlackLitterman.md) · [RobustOptimization](RobustOptimization.md) · [PortfolioConstraints](PortfolioConstraints.md)

---

## Formulations

### Mean–variance

\[
\max_w \; w^\top \mu - \frac{\lambda}{2}\, w^\top \Sigma w
\quad\text{s.t.}\quad
\mathbf{1}^\top w = b,\;\;
\ell \le w_i \le u,\;\;
\text{(optional) }\sum_i |w_i| \le G
\]

Implemented as projected gradient / SciPy minimize with ridge-stabilized \(\Sigma\). Hard constraints are never silently relaxed; infeasible specs return `success=False`.

### Minimum variance

\[
\min_w \; w^\top \Sigma w
\quad\text{s.t.}\quad
\text{same budget / box / gross}
\]

Analytic inverse-\(\Sigma\) solution when unconstrained, then projected onto the feasible set. \(\mu\) is ignored (signature-compatible only).

### Maximum Sharpe

\[
\max_w \; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
\]

Uses excess returns and numerically stable negative-Sharpe minimization. Requires \(\mu\).

---

## \(\mu\) shrinkage and winsorization

Before return-seeking solves, `stabilize_mu` (in `optimization.projection`) winsorizes by MAD \(z\)-score (`winsor_z=3`) and clips to `±mu_clip` (default `0.5`) to avoid extreme corner allocations from noisy forecasts.

James–Stein / grand-mean shrinkage for expected returns lives in `iqrp.app.portfolio.expected_returns`:

```python
from iqrp.app.portfolio.expected_returns import shrinkage_expected_returns, james_stein_shrinkage

er = shrinkage_expected_returns(returns, names=names)
# or james_stein_shrinkage(mu, cov=Sigma, n_obs=T)
```

Engine path: `eng.expected_returns(returns=R, method="shrinkage")`. Forecast confidence shrink (separate from JS) uses `forecast_expected_returns` — low confidence pulls toward prior; confidence never expands certainty.

---

## Constraints

Parsed by `parse_constraints` / `check_feasibility`:

| Parameter | Role |
|-----------|------|
| `long_only` | Lower bound 0 |
| `max_weight` / `min_weight` | Box |
| `max_gross` | Cap on \(\sum |w_i|\) |
| `budget` | \(\sum w_i = b\) (default 1) |
| `constraints` | Dict override of the above |

On conflict (e.g. `n * max_weight < budget`), result is infeasible with `conflicting_constraints` — no auto-relaxation.

---

## Examples

```python
import numpy as np
from iqrp.app.portfolio.optimization import (
    optimize_mean_variance,
    optimize_minimum_variance,
    optimize_maximum_sharpe,
)

mu = np.array([0.10, 0.07, 0.05, 0.03])
cov = np.array([
    [0.04, 0.01, 0.01, 0.00],
    [0.01, 0.09, 0.02, 0.01],
    [0.01, 0.02, 0.16, 0.02],
    [0.00, 0.01, 0.02, 0.25],
])
names = ["eq", "fi", "cmd", "cash_proxy"]

mv = optimize_mean_variance(
    mu=mu, cov=cov, risk_aversion=2.0,
    long_only=True, max_weight=0.4, names=names, mu_clip=0.5,
)
gmv = optimize_minimum_variance(cov=cov, long_only=True, max_weight=0.5, names=names)
tan = optimize_maximum_sharpe(
    mu=mu, cov=cov, risk_free_rate=0.02,
    long_only=True, max_weight=0.4, names=names,
)

mv["success"], mv["weights"], mv.get("diagnostics")
```

Via the engine:

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
opt = eng.optimize(mu=mu, cov=cov, method="mean_variance", names=names)
opt2 = eng.optimize(mu=mu, cov=cov, method="max_sharpe", names=names)
opt3 = eng.optimize(cov=cov, method="min_variance", names=names)
```

Each dict / `OptimizationResult` exposes `success`, `weights`, `status`, `failure_reason`, `conflicting_constraints`, `expected_return`, `expected_variance`, and `diagnostics`.
