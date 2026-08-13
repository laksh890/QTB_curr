# Portfolio Construction

Institutional portfolio construction: express provided forecasts and signals under hard constraints, costs, and Risk Intelligence gates.

Package: `iqrp.app.portfolio`  
Primary type: `PortfolioConstructionEngine`  
Hydra config: `iqrp/configs/portfolio/default.yaml`

Related: [MeanVariance](MeanVariance.md) · [RiskParity](RiskParity.md) · [BlackLitterman](BlackLitterman.md) · [RobustOptimization](RobustOptimization.md) · [TransactionCosts](TransactionCosts.md) · [TurnoverControl](TurnoverControl.md) · [MultiPeriodOptimization](MultiPeriodOptimization.md) · [PortfolioConstraints](PortfolioConstraints.md) · [Phase 10 summary](Phase10_PortfolioConstruction.md)

---

## Placement

```text
Forecast Intelligence (μ, confidence)     Risk Intelligence (limits / decisions)
Capital Allocation (strategy budgets)              │
        │                                          │
        └──────────────┬───────────────────────────┘
                       ▼
           PortfolioConstructionEngine
                       │
                       ▼
     weights · positions · costs · ValidationReport · audit
```

Construction **consumes** Forecast, Risk, and Capital modules via imports only. It never invents alpha from history alone and never reimplements sizing / hierarchical capital solvers.

---

## Architectural rules

| # | Rule |
|---|------|
| 1 | **No alpha generation.** Only expresses caller-supplied forecasts/signals under constraints. |
| 2 | **Hard constraints never silently relaxed.** Infeasible or failed optimizations do not widen bounds. |
| 3 | **Explicit fallback.** On failure, apply configured `current` \| `min_variance` \| `cash` with `fallback_used=True`. |
| 4 | **Risk Intelligence final authority** when `require_risk_validation` is true. Reject → fallback. |
| 5 | **Confidence cannot invent certainty.** Forecast confidence only shrinks toward prior; it never overrides hard limits. |
| 6 | **Transaction costs when configured.** `construct` / `transaction_cost` include commission, spread, slippage, impact. |
| 7 | **Point-in-time only.** No look-ahead; estimators and paths use information available at decision time. |
| 8 | **Full audit trail.** Every `PortfolioResult` / `OptimizationResult` records method, versions, seed, fallback, risk gate. |
| 9 | **Import-only integration.** Forecast, Risk, and Capital Allocation are dependencies — not rewritten here. |
| 10 | **Soft vs hard.** Soft violations are reported; hard violations invalidate the book and never auto-relax. |
| 11 | **Infeasible → `success=False`.** Optimizers return failure with `conflicting_constraints`; engine then falls back. |
| 12 | **Reproducibility.** `seed`, `data_version`, and `model_version` travel with settings and results. |

---

## Quick start

```python
import numpy as np
from iqrp.app.portfolio import PortfolioConstructionEngine, PortfolioSettings

eng = PortfolioConstructionEngine(PortfolioSettings.default())
mu = np.array([0.08, 0.06, 0.04])
cov = np.diag([0.04, 0.09, 0.16])
opt = eng.optimize(mu=mu, cov=cov, method="mean_variance", names=["a", "b", "c"])

out = eng.construct(
    forecasts=mu,
    returns=np.random.randn(252, 3) * 0.01,
    capital=1e6,
    prices=np.array([100.0, 50.0, 25.0]),
    names=["a", "b", "c"],
)
out.weights, out.turnover, out.fallback_used
```

---

## Hydra config

File: `iqrp/configs/portfolio/default.yaml`

| Key | Default | Role |
|-----|---------|------|
| `method` | `mean_variance` | Default optimizer key |
| `long_only` | `true` | Non-negative weights |
| `max_weight` | `0.4` | Per-name box |
| `max_gross` / `max_leverage` | `1.5` / `2.0` | Exposure caps |
| `max_turnover` | `0.5` | One-way turnover hard cap when used |
| `risk_aversion` | `1.0` | MV / robust λ |
| `fallback` | `current` | `current` \| `min_variance` \| `cash` |
| `require_risk_validation` | `true` | Gate via Risk Intelligence |
| `covariance.method` | `shrinkage` | `sample` \| `ewma` \| `shrinkage` \| `ledoit_wolf` \| `factor` \| `robust` |
| `expected_returns.method` | `forecast` | `forecast` \| `historical` \| `shrinkage` \| `black_litterman` |
| `expected_returns.confidence_shrink` | `true` | Shrink forecasts by confidence |
| `objective.turnover_penalty` | `0.0` | Soft TC in turnover-aware opt |
| `objective.cvar_confidence` | `0.95` | CVaR / ES level |

```python
from iqrp.app.portfolio import PortfolioSettings

settings = PortfolioSettings.from_hydra(overrides=["method=risk_parity", "fallback=cash"])
eng = PortfolioConstructionEngine(settings)
```

---

## `PortfolioConstructionEngine` API

### `optimize`

Dispatch to `iqrp.app.portfolio.optimization.*` and return `OptimizationResult`.

```python
res = eng.optimize(
    mu=mu, cov=cov, method="max_sharpe",
    names=["a", "b", "c"], current_weights=w0,
    long_only=True, max_weight=0.35, risk_aversion=1.0,
)
```

| Method key | Backend |
|------------|---------|
| `mean_variance` | `optimize_mean_variance` |
| `min_variance` / `minimum_variance` | `optimize_minimum_variance` |
| `max_sharpe` / `maximum_sharpe` | `optimize_maximum_sharpe` |
| `max_diversification` | `optimize_maximum_diversification` |
| `risk_parity` / `erc` | `optimize_risk_parity` |
| `hrp` / `herc` | `optimize_hrp` / `optimize_herc` |
| `min_cvar` / `cvar` | `optimize_cvar` |
| `drawdown` | `optimize_drawdown` |
| `turnover` / `turnover_aware` | `optimize_turnover` |
| `robust` | `optimize_robust` |
| `black_litterman` | `optimize_black_litterman` |
| `entropy` | `optimize_entropy` |

If `cov` is omitted and `returns` is provided, covariance is estimated via `covariance()`. For return-seeking methods, missing `mu` is filled from `expected_returns()`.

### `construct`

End-to-end: forecasts/signals → μ → cov → optimize → risk gate → target weights/positions → costs & diagnostics.

```python
result = eng.construct(
    forecasts=mu,
    forecast_confidence=np.array([0.9, 0.7, 0.5]),
    returns=R,
    current_portfolio=w0,
    capital=1_000_000,
    prices=prices,
    method="mean_variance",
    include_transaction_costs=True,
    adv=[5e6, 3e6, 2e6],
    spreads=[0.0005, 0.0008, 0.001],
)
```

Alternate signal path (no cov): `signal_method="zscore"` maps signals → raw weights without inventing α.

### `target_weights` / `target_positions`

```python
tw = eng.target_weights(weights=[0.4, 0.3, 0.3], names=["a", "b", "c"])
tw = eng.target_weights(signals=s, method="zscore", names=names)
tp = eng.target_positions(tw, capital=1e6, prices=prices)
```

### `expected_returns`

| Method | Function |
|--------|----------|
| `forecast` | `forecast_expected_returns` (confidence shrink toward prior) |
| `historical` | `historical_expected_returns` |
| `shrinkage` | `shrinkage_expected_returns` (James–Stein) |
| `black_litterman` / `bl` | `black_litterman_posterior` |

```python
er = eng.expected_returns(forecasts=mu, confidence=conf, method="forecast")
# er["mu"] / er["vector"]
```

### `covariance`

| Method | Function |
|--------|----------|
| `sample` | `sample_covariance` |
| `ewma` | `ewma_covariance` (`ewma_lambda`) |
| `shrinkage` | `shrinkage_covariance` |
| `ledoit_wolf` | `ledoit_wolf_covariance` |
| `factor` | `factor_covariance` |
| `robust` | `robust_covariance` (winsorize / MCD-style) |

```python
c = eng.covariance(returns=R, method="shrinkage")
Sigma = c["matrix"]
```

### `risk_contribution`

```python
rc = eng.risk_contribution(weights=w, cov=Sigma)
# rc includes per-name contributions / shares from portfolio_risk
```

### `rebalance`

Wraps `plan_rebalance` with optional bands (no-trade region).

```python
plan = eng.rebalance(w0, w_target, absolute_band=0.02, relative_band=0.1, min_trade=0.005)
plan.should_rebalance, plan.trades, plan.turnover
```

### `validate`

Constraint checks via `check_all_constraints` plus optional Risk Intelligence pre-trade.

```python
report = eng.validate(
    w, max_weight=0.4, max_gross=1.5, long_only=True,
    returns=R, forecast_confidence=conf,
)
report.valid, report.hard_violations, report.risk_decision
```

### `transaction_cost` / `turnover` / `diagnostics`

```python
tc = eng.transaction_cost(w0, w1, capital=1e6, prices=prices, adv=adv, spreads=spreads)
to = eng.turnover(w0, w1)          # {"turnover", "one_way", "two_way"}
diag = eng.diagnostics(w, cov=Sigma, mu=mu)
```

### `save` / `load`

```python
eng.save("/tmp/portfolio_state.json", obj=result)
payload = eng.load("/tmp/portfolio_state.json")  # restores settings when present
```

---

## `PortfolioResult` fields

Returned by `construct`:

| Field | Meaning |
|-------|---------|
| `portfolio_weights` | `TargetWeights` |
| `target_positions` | `TargetPositions` / position list |
| `weights` / `names` | Flat weight vector and labels |
| `expected_return` / `expected_volatility` / `expected_sharpe` | From μ, Σ, risk-free |
| `expected_cvar` / `expected_drawdown` | Optional scenario / path metrics |
| `gross_exposure` / `net_exposure` | Σ\|w\| and Σw |
| `turnover` | One-way vs current |
| `transaction_cost` | Structured cost dict |
| `risk_contribution` | Marginal / component risk |
| `factor_exposure` / `liquidity_exposure` | When loadings / ADV supplied |
| `optimization` | Nested `OptimizationResult` |
| `fallback_used` / `fallback_kind` / `fallback_reasons` | Explicit fallback audit |
| `risk_validation` | Pre-trade decision payload |
| `constraints` | Applied box / leverage audit |
| `data_version` / `model_version` / `seed` | Reproducibility |
| `audit` | μ meta, cov method, risk skip reason |
| `success` / `status` / `messages` | Outcome |

`ValidationReport`: `valid`, `violations`, `hard_violations`, `soft_violations`, `risk_decision`, `risk_breaches`, `messages`, `timestamp`, `meta`.

---

## Fallback behavior

On optimizer exception, `success=False`, or Risk rejection:

1. Read `settings.fallback`.
2. **`current`** — keep `current_weights` if available (`fallback_kind="current"`).
3. **`min_variance`** — solve GMV under the same hard box/budget; if that fails, continue.
4. **`cash`** — all-zero weights (also used when current unavailable).

Never widen `max_weight`, turnover caps, or long-only. Always set `fallback_used=True` and record reasons in diagnostics / audit.

---

## `require_risk_validation`

When `true` (default):

1. Engine constructs or accepts `RiskIntelligenceEngine` / ensemble.
2. `construct` / `validate` call `_run_risk_validation` (`check_limits`, `validate_position`).
3. Actions `REJECT` / `HALT` / `BLOCK` or `approved=False` trigger fallback.
4. If the risk engine is unavailable, validation is **skipped with a warning** (`action=SKIP`, `approved=True`) — construction proceeds but `audit.risk_skip_reason` is set. Soft limit breaches alone yield `CAUTION` without forcing fallback.

Hard limit breaches from `check_limits` always reject.

---

## Integration (import-only)

| Upstream | How used |
|----------|----------|
| Forecast Intelligence | `forecasts` / confidence → `forecast_expected_returns`; never backfills α from unlabeled history as “forecast” without an explicit historical method |
| Risk Intelligence | Pre-trade `validate_position` / `check_limits` when `require_risk_validation` |
| Capital Allocation | Risk-parity / HRP / HERC backends live in `iqrp.app.risk.sizing` and `iqrp.app.risk.capital.hierarchical`; portfolio wrappers import them |

```python
from iqrp.app.portfolio import PortfolioConstructionEngine
from iqrp.app.risk import RiskIntelligenceEngine, RiskIntelligenceEnsemble

risk = RiskIntelligenceEngine()
eng = PortfolioConstructionEngine(risk_engine=risk, risk_ensemble=RiskIntelligenceEnsemble(risk_engine=risk))
```

---

## Validation

```bash
python -m iqrp.app.portfolio.phase10
# or
from iqrp.app.portfolio import validate_phase10, write_phase10_report
```

Machine-readable report: [Phase10_PortfolioConstruction_Validation.json](Phase10_PortfolioConstruction_Validation.json).
