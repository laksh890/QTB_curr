# Portfolio Constraints

Constraint modules for exposure, leverage, concentration, liquidity, turnover, sector, factor, currency, beta, risk, and position limits. Aggregator: `check_all_constraints`.

Package: `iqrp.app.portfolio.constraints`  
Facade: `PortfolioConstructionEngine.validate`

Related: [PortfolioConstruction](PortfolioConstruction.md) · [TurnoverControl](TurnoverControl.md) · [RiskLimits](RiskLimits.md)

---

## Hard constraints are never silently relaxed

- Checkers **only report** `ConstraintViolation` objects; they do not mutate weights or widen limits.
- Optimizers that cannot satisfy hard box/budget/turnover return `success=False` with `conflicting_constraints`.
- Soft severity is explicit (`severity`, `hard=False`, or `soft_constraints` name prefixes). Soft may be ignored by callers; hard violations invalidate `ValidationReport.valid`.
- Forecast confidence never overrides hard portfolio or risk limits.

---

## Modules

| Module | Checker | Typical kwargs |
|--------|---------|----------------|
| `exposure` | `check_exposure_constraints` | `max_gross`, `max_net`, `min_net`, `max_long`, `max_short` |
| `leverage` | `check_leverage_constraints` | `max_leverage`, `min_leverage` |
| `concentration` | `check_concentration_constraints` | `max_weight`, `max_hhi`, `min_effective_n` |
| `position` | `check_position_constraints` | `max_position`, `min_position`, `min_weight`, `long_only` |
| `liquidity` | `check_liquidity_constraints` | `adv`, `max_participation`, `max_ttl`, `min_adv_coverage` |
| `turnover` | `check_turnover_constraints` | `current_weights` / `weights_old`, `max_turnover`, `min_trade` |
| `sector` | `check_sector_constraints` | `sector_map`, `max_sector_weight`, `min_sector_weight` |
| `factor` | `check_factor_constraints` | `factor_loadings`, factor min/max / net exposures |
| `currency` | `check_currency_constraints` | currency map / net FX caps |
| `beta` | `check_beta_constraints` | `betas`, `max_beta`, `min_beta`, `target_beta` |
| `risk` | `check_risk_constraints` | portfolio vol / VaR / CVaR style caps when provided |

Only constraints with explicit limit kwargs (or required side inputs like `adv` + participation) are evaluated.

---

## `check_all_constraints`

```python
import numpy as np
from iqrp.app.portfolio.constraints import check_all_constraints, ConstraintSeverity

w = np.array([0.5, 0.3, 0.2])
violations = check_all_constraints(
    w,
    max_weight=0.4,
    max_gross=1.5,
    max_leverage=2.0,
    long_only=True,
    max_turnover=0.2,
    current_weights=np.array([0.3, 0.4, 0.3]),
    adv=np.array([5e6, 3e6, 2e6]),
    capital=1e6,
    max_participation=0.1,
    soft_constraints=["max_hhi"],  # treat matching names as soft
)
hard = [v for v in violations if v.hard]
```

Each `ConstraintViolation` carries name, message, severity, observed vs limit, and optional metadata (`to_dict()`).

---

## Engine validation

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
report = eng.validate(
    w,
    max_weight=0.4,
    max_gross=1.5,
    long_only=True,
    returns=R,
    forecast_confidence=conf,
    risk_validation=True,
)
report.valid, report.hard_violations, report.soft_violations, report.risk_decision
```

`valid` requires **no hard violations** and Risk Intelligence approval when risk validation is enabled (`approved` is not `False`).

---

## Helpers

```python
from iqrp.app.portfolio.constraints import (
    exposure_metrics,
    leverage_metrics,
    concentration_metrics,
    currency_exposures,
    sector_exposures,
    portfolio_factor_exposures,
    portfolio_beta,
    turnover,
)

exposure_metrics(w)
concentration_metrics(w)
turnover(w0, w1)
```
