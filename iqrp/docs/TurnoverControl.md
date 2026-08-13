# Turnover Control

Hard and soft turnover limits, rebalance bands / no-trade regions, turnover-aware optimization, and `plan_rebalance` triggers.

Package: `iqrp.app.portfolio.optimization` · `iqrp.app.portfolio.construction` · `iqrp.app.portfolio.constraints`  
Entry points: `optimize_turnover`, `plan_rebalance`, `check_turnover_constraints`  
Facade: `PortfolioConstructionEngine.rebalance` / `optimize(..., method="turnover")`

Related: [TransactionCosts](TransactionCosts.md) · [MultiPeriodOptimization](MultiPeriodOptimization.md) · [PortfolioConstraints](PortfolioConstraints.md)

---

## Hard vs soft turnover

| Mode | Mechanism | Failure |
|------|-----------|---------|
| **Hard** | `max_turnover` on `optimize_turnover` or `check_turnover_constraints` | Infeasible / violation reported — **never silently relaxed** |
| **Soft** | `turnover_penalty` \(\tau\) in the MV objective | Prefer low turnover; may still trade if α justifies |

One-way turnover: \(\tfrac12 \sum_i |w_i - w_{0,i}|\). Engine `turnover()` also reports `two_way = 2 × one_way`.

```python
from iqrp.app.portfolio.constraints import check_turnover_constraints, turnover

to = turnover(w0, w1)
violations = check_turnover_constraints(w1, current_weights=w0, max_turnover=0.25)
```

---

## Turnover-aware optimization

\[
\max_w \; w^\top\mu - \tfrac{\lambda}{2} w^\top\Sigma w - \tau\,\|w-w_0\|_1
\]

with optional hard \( \tfrac12 \|w-w_0\|_1 \le T_{\max} \).

```python
import numpy as np
from iqrp.app.portfolio.optimization import optimize_turnover

mu = np.array([0.08, 0.05, 0.03])
cov = np.diag([0.04, 0.09, 0.16])
w0 = np.array([0.4, 0.3, 0.3])

out = optimize_turnover(
    mu=mu, cov=cov, current_weights=w0,
    risk_aversion=1.0, turnover_penalty=0.02, max_turnover=0.15,
    long_only=True, max_weight=0.45, names=["a", "b", "c"],
)
```

Engine: `method="turnover"` or `"turnover_aware"`; defaults from `objective.turnover_penalty` / `max_turnover` in Hydra.

---

## Bands and no-trade region

`RebalanceBands` suppress trades inside absolute / relative bands and below `min_trade`:

```python
from iqrp.app.portfolio.construction import RebalanceBands, plan_rebalance

bands = RebalanceBands(absolute=0.02, relative=0.10, min_trade=0.005)
plan = plan_rebalance(w0, w_target, bands=bands, names=["a", "b", "c"])
plan.should_rebalance, plan.trades, plan.turnover
```

Via engine:

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
plan = eng.rebalance(w0, w_target, absolute_band=0.02, relative_band=0.1, min_trade=0.005)
```

---

## `plan_rebalance` triggers

`RebalanceTrigger.kind` values: `scheduled`, `threshold`, `risk`, `drift`, `regime`, `drawdown`, `liquidity`, `manual`.

```python
plan = plan_rebalance(
    w0, w_target,
    bands=bands,
    force=False,
    scheduled=True,
    turnover_threshold=0.05,
    drift_threshold=0.05,
    risk_breach=False,
    drawdown=-0.12, drawdown_threshold=-0.10,
)
[t for t in plan.triggers if t.fired]
```

`RebalancePlan` includes `should_rebalance`, `triggers`, current/target weights, trades, turnover, and audit metadata. Dynamic rebalancing for Phase 10 validation maps to this API (`construction.plan_rebalance`).
