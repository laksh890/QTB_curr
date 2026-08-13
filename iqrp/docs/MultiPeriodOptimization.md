# Multi-Period Optimization

Horizon-aware portfolio paths with drift between rebalances, transaction costs across dates, and a small-state DP / greedy heuristic.

Package: `iqrp.app.portfolio.multi_period`  
Entry points: `optimize_multi_period`, `optimize_dynamic_programming`, `apply_drift`, `rebalance_schedule`

Related: [TurnoverControl](TurnoverControl.md) · [TransactionCosts](TransactionCosts.md) · [MeanVariance](MeanVariance.md) · [PortfolioConstruction](PortfolioConstruction.md)

---

## `optimize_multi_period`

At each rebalance date \(t\), solve single-period mean–variance on \((\mu_t, \Sigma_t)\), penalizing turnover from the **drifted** book. Between rebalances, weights evolve with `return_path` (or expected \(\mu\)).

```python
import numpy as np
from iqrp.app.portfolio.multi_period import optimize_multi_period

mu = np.array([0.06, 0.05, 0.04])
cov = np.diag([0.04, 0.09, 0.16])
w0 = np.array([1/3, 1/3, 1/3])

# Optional time-varying forecasts
mu_path = np.vstack([mu, mu * 1.05, mu * 0.95])
return_path = np.random.randn(3, 3) * 0.01

out = optimize_multi_period(
    mu=mu, cov=cov,
    current_weights=w0,
    horizons=3,
    mu_path=mu_path,
    return_path=return_path,
    transaction_cost=0.001,
    rebalance_every=1,
    turnover_threshold=None,
    long_only=True, max_weight=0.5,
    names=["a", "b", "c"],
)
out["success"], out["weights"], out.get("path")  # terminal + horizon path / diagnostics
```

| Parameter | Role |
|-----------|------|
| `horizons` | Number of decision periods |
| `mu_path` / `cov_path` | Per-horizon forecasts (tiled from single μ/Σ if omitted) |
| `return_path` | Realized/expected returns for drift between rebalances |
| `transaction_cost` | Linear TC × turnover at rebalance dates |
| `rebalance_every` | Rebalance cadence in horizon steps |
| `turnover_threshold` | Skip rebalance when turnover below threshold |

Hard constraints (`long_only`, `max_weight`, `max_gross`, `budget`) bind at each solve; infeasible horizons surface as failure — no silent relaxation.

---

## Drift

```python
from iqrp.app.portfolio.multi_period import apply_drift

w_drifted = apply_drift(w0, returns_row)  # renormalize after asset returns
```

Between rebalances the book is not free to jump; TC is charged only when `rebalance_schedule` fires.

---

## DP heuristic

`optimize_dynamic_programming` runs backward DP on a discrete simplex grid for **small** \(n\) and horizons. When the combinatorial cost is large (\(n>6\), \(h>4\), or dense grids), it switches to a **greedy heuristic** while keeping the same objective structure:

\[
\sum_t \Big( w_t^\top \mu_t - \tfrac{\lambda}{2} w_t^\top \Sigma w_t - c\cdot \mathrm{turnover}(w_{t-1}^{\mathrm{drift}}, w_t) \Big)
\]

```python
from iqrp.app.portfolio.multi_period import optimize_dynamic_programming

dp = optimize_dynamic_programming(
    mu=mu, cov=cov, current_weights=w0,
    horizons=2, grid_levels=4,
    transaction_cost=0.001,
    risk_aversion=1.0, long_only=True, max_weight=0.5,
)
```

---

## TC across horizons

Each rebalance pays `transaction_cost * turnover(drifted, target)`. Cumulative cost and per-horizon weights are recorded in diagnostics so multi-period plans remain auditable and point-in-time (no future path leakage beyond caller-supplied `mu_path` / `return_path`).
