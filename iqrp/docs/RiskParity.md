# Risk Parity

Risk parity, equal risk contribution (ERC), hierarchical risk parity (HRP), and hierarchical ERC (HERC).

Package: `iqrp.app.portfolio.optimization`  
Entry points: `optimize_risk_parity`, `optimize_hrp`, `optimize_herc`, `optimize_hierarchical`

Related: [PortfolioConstruction](PortfolioConstruction.md) · [CapitalAllocation](CapitalAllocation.md) · [PositionSizing](PositionSizing.md)

---

## Design note — import-only backends

These optimizers **do not reimplement** the core solvers:

| Portfolio API | Delegates to |
|---------------|--------------|
| `optimize_risk_parity` / ERC | `iqrp.app.risk.sizing.risk_parity.risk_parity_weights` / `equal_risk_contribution` |
| `optimize_hrp` | `iqrp.app.risk.capital.hierarchical.hrp_weights` |
| `optimize_herc` | `iqrp.app.risk.capital.hierarchical.herc_weights` |

After the risk/capital backend returns weights, the portfolio layer scales to `budget`, projects onto hard box/gross constraints, and returns **infeasible** if projection cannot satisfy hard limits (never silently relaxes them). Risk parity / hierarchical methods require non-negative weights; short-capable specs are rejected as infeasible.

---

## Risk parity / ERC

Equalize (or target) risk contributions \(w_i (\Sigma w)_i\).

```python
import numpy as np
from iqrp.app.portfolio.optimization import optimize_risk_parity

cov = np.diag([0.04, 0.09, 0.16])
rp = optimize_risk_parity(cov=cov, method="risk_parity", long_only=True, max_weight=0.5, names=["a", "b", "c"])
erc = optimize_risk_parity(cov=cov, method="erc", long_only=True, max_weight=0.5)
```

Engine aliases: `method="risk_parity"` or `method="erc"` (also `risk_budget` / `equal_risk` → risk parity).

---

## HRP / HERC

Hierarchical clustering on the correlation structure, then recursive bisection / equal-risk allocation within clusters.

```python
from iqrp.app.portfolio.optimization import optimize_hrp, optimize_herc

hrp = optimize_hrp(cov=cov, linkage="single", long_only=True, max_weight=0.45, names=["a", "b", "c"])
herc = optimize_herc(cov=cov, linkage="single", long_only=True, max_weight=0.45, names=["a", "b", "c"])
```

Optional `corr=` overrides the correlation implied by `cov`. `optimize_hierarchical(..., variant="hrp"|"herc")` is the shared entrypoint.

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
eng.optimize(cov=cov, method="hrp", names=["a", "b", "c"])
eng.optimize(cov=cov, method="herc", names=["a", "b", "c"])
```

---

## Risk contribution diagnostics

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
w = eng.optimize(cov=cov, method="erc", names=["a", "b", "c"]).weights
rc = eng.risk_contribution(w, cov)
```

---

## Constraints and failure modes

- Hard `max_weight` / `max_gross` / `budget` bind after projection.
- If projected weights violate box bounds beyond tolerance → `success=False`, `conflicting_constraints` populated.
- Engine fallback (`current` \| `min_variance` \| `cash`) applies only at the facade layer — not inside these functions.
