# Phase 10 — Portfolio Construction

Institutional portfolio construction for Forecast → Risk → Portfolio: express provided forecasts/signals under hard constraints, costs, turnover control, and Risk Intelligence pre-trade gates. Does **not** generate alpha.

Machine-readable validation: [`Phase10_PortfolioConstruction_Validation.json`](Phase10_PortfolioConstruction_Validation.json)

---

## Completed components

| Component | Location | Docs |
|-----------|----------|------|
| Portfolio Construction Framework | `iqrp.app.portfolio.PortfolioConstructionEngine` | [PortfolioConstruction.md](PortfolioConstruction.md) |
| Expected Return Engine | `iqrp.app.portfolio.expected_returns` | [PortfolioConstruction.md](PortfolioConstruction.md) |
| Covariance Engine | `iqrp.app.portfolio.covariance` | [PortfolioConstruction.md](PortfolioConstruction.md) |
| Mean–Variance / Min Var / Max Sharpe | `optimization.optimize_mean_variance` / `minimum_variance` / `maximum_sharpe` | [MeanVariance.md](MeanVariance.md) |
| Risk Parity / ERC | `optimization.optimize_risk_parity` | [RiskParity.md](RiskParity.md) |
| HRP / HERC | `optimization.optimize_hrp` / `optimize_herc` | [RiskParity.md](RiskParity.md) |
| Max Diversification / CVaR / Drawdown | `optimize_maximum_diversification` / `cvar` / `drawdown` | [PortfolioConstruction.md](PortfolioConstruction.md) |
| Black–Litterman | `optimization.optimize_black_litterman` | [BlackLitterman.md](BlackLitterman.md) |
| Robust Optimization | `optimization.optimize_robust` | [RobustOptimization.md](RobustOptimization.md) |
| Transaction Costs | `transaction_costs.total_transaction_cost` | [TransactionCosts.md](TransactionCosts.md) |
| Turnover Control | `optimization.optimize_turnover` · `construction.plan_rebalance` | [TurnoverControl.md](TurnoverControl.md) |
| Multi-Period | `multi_period.optimize_multi_period` | [MultiPeriodOptimization.md](MultiPeriodOptimization.md) |
| Constraints / Liquidity / Factor / FX | `constraints.check_all_constraints` (+ liquidity, factor, currency) | [PortfolioConstraints.md](PortfolioConstraints.md) |
| Validation + Risk pre-trade | `ValidationReport` · `require_risk_validation` | [PortfolioConstruction.md](PortfolioConstruction.md) |

---

## Architectural rules (summary)

1. No alpha generation — express forecasts/signals only.  
2. Hard constraints never silently relaxed.  
3. Explicit fallback (`current` \| `min_variance` \| `cash`) with `fallback_used=True`.  
4. Risk Intelligence final authority when `require_risk_validation`.  
5. Forecast confidence cannot invent certainty or override hard limits.  
6. Transaction costs included when configured.  
7. Point-in-time only.  
8. Auditable, reproducible decisions (`seed`, `data_version`, `model_version`).  

Full 12-rule table: [PortfolioConstruction.md](PortfolioConstruction.md#architectural-rules).

---

## Integration hooks (import-only)

**Package:** `iqrp/app/portfolio/__init__.py`  
Exports `PortfolioConstructionEngine`, `PortfolioSettings`, `Portfolio`, `OptimizationResult`, construction types, `validate_phase10`.

**Hydra:** `iqrp/configs/portfolio/default.yaml`

**Upstream (import-only):** Forecast Intelligence (μ / confidence), Risk Intelligence (pre-trade), Capital Allocation / risk sizing (RP, HRP, HERC backends). No reimplementation of those solvers inside portfolio.

---

## Canonical usage

```python
import numpy as np
from iqrp.app.portfolio import PortfolioConstructionEngine, PortfolioSettings

eng = PortfolioConstructionEngine(PortfolioSettings.default())
mu = np.array([0.08, 0.06, 0.04])
cov = np.diag([0.04, 0.09, 0.16])
R = np.random.randn(252, 3) * 0.01

opt = eng.optimize(mu=mu, cov=cov, method="mean_variance", names=["a", "b", "c"])
out = eng.construct(
    forecasts=mu, returns=R, capital=1e6,
    prices=np.array([100.0, 50.0, 25.0]), names=["a", "b", "c"],
)
report = eng.validate(out.weights, max_weight=0.4, long_only=True, returns=R)
```

---

## Documentation index

- [PortfolioConstruction.md](PortfolioConstruction.md) — engine API, `PortfolioResult`, Hydra, fallback, risk gate  
- [MeanVariance.md](MeanVariance.md) — MV / GMV / max Sharpe, μ stabilization  
- [RiskParity.md](RiskParity.md) — RP, ERC, HRP, HERC (risk/capital backends)  
- [BlackLitterman.md](BlackLitterman.md) — equilibrium, views, posterior, multi-forecast  
- [RobustOptimization.md](RobustOptimization.md) — uncertainty sets, DRO, parameter uncertainty  
- [TransactionCosts.md](TransactionCosts.md) — commission, spread, slippage, impact  
- [TurnoverControl.md](TurnoverControl.md) — hard/soft turnover, bands, `plan_rebalance`  
- [MultiPeriodOptimization.md](MultiPeriodOptimization.md) — horizons, drift, DP heuristic  
- [PortfolioConstraints.md](PortfolioConstraints.md) — all checkers, `check_all_constraints`  

---

## Validation

```bash
python -m iqrp.app.portfolio.phase10
```

```python
from iqrp.app.portfolio import validate_phase10, write_phase10_report

report = validate_phase10()
path = write_phase10_report()  # writes Phase10_PortfolioConstruction_Validation.json
```

Status and component checklist live in the JSON report (`status`, `summary`, `checklist`, `components`, `architectural_rules`).
