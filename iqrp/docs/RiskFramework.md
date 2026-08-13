# Institutional Risk Intelligence Framework

Central risk architecture for IQRP. Sits **between Alpha / Forecasting and Portfolio Construction / Execution**. No trading strategy may bypass this layer.

Package: `iqrp.app.risk`  
Hydra config: `iqrp/configs/risk/default.yaml`

Related: [VaR](VaR.md) · [Expected Shortfall](ExpectedShortfall.md) · [Monte Carlo Risk](MonteCarloRisk.md) · [Position Sizing](PositionSizing.md) · [Risk Limits](RiskLimits.md) · [Stress Testing](StressTesting.md) · [Liquidity Risk](LiquidityRisk.md) · [Model Risk](ModelRisk.md) · [Drawdown Control](DrawdownControl.md) · [Volatility Risk Integration](RiskIntegration.md)

---

## Placement in the stack

```text
Forecast Intelligence / Volatility / Regimes / Features / Probability
                         │
                         ▼  (import-only contracts; Risk never writes back)
              ┌──────────────────────────┐
              │  RiskIntelligenceEngine  │  ← sole pre-trade gate
              └──────────────────────────┘
                         │
                         ▼
              Portfolio Construction / Execution / Live
```

Risk **consumes** upstream intelligence and **emits** risk measures, limit decisions, and audit records. It does not generate alpha, alter market data, or place orders.

---

## Ten architectural rules

| # | Rule |
|---|------|
| 1 | **Risk never generates alpha.** No signal invention, ranking, or strategy logic inside `iqrp.app.risk`. |
| 2 | **Risk never modifies historical data.** Inputs are read-only views; transforms are local copies. |
| 3 | **Risk calculations are point-in-time correct.** Only information available as of the decision timestamp may be used. |
| 4 | **No future information** may enter risk calculations (no look-ahead in EWMA, FHS, stress windows, or model-risk windows). |
| 5 | **Hard risk limits cannot be overridden by forecasting confidence.** |
| 6 | **High forecast confidence never authorizes unlimited leverage.** Confidence is capped (`confidence_cap`). |
| 7 | **Every rejection has an explicit reason** on `RiskDecision.reason` and in the audit log. |
| 8 | **Decisions are reproducible** from recorded inputs, parameters, `data_version`, and `model_version`. |
| 9 | **No trading component may bypass** `validate_position()` / `check_limits()`. |
| 10 | **Separate research metrics from live trading risk metrics.** Research notebooks may explore; live path uses `RiskIntelligenceEngine` + Hydra defaults. |

---

## Package layout

```text
iqrp/app/risk/
  base/           # RiskMeasure, RiskReport, RiskLimit, RiskState, RiskDecision
  portfolio/      # exposure, concentration, factor, diversification, portfolio vol
  market/         # volatility, beta, correlation, liquidity, gap
  tail/           # VaR, CVaR, ES, drawdown, tail dependence
  stress/         # historical, hypothetical, reverse, scenarios
  simulation/     # parametric MC, bootstrap, block bootstrap, copula, ScenarioEngine
  sizing/         # vol target, Kelly, fractional Kelly, risk parity, DD-adjusted
  leverage/       # dynamic leverage + hard clips
  limits/         # position, exposure, loss, concentration, liquidity
  model_risk/     # disagreement, uncertainty, drift, parameter uncertainty
  monitoring/     # monitor, alerts, breaches, dashboards
  aggregation/    # flat + hierarchical aggregation
  orchestrator.py # RiskIntelligenceEngine
  config.py       # RiskSettings (Pydantic + Hydra)
```

---

## Configuration

```python
from iqrp.app.risk import RiskSettings, RiskIntelligenceEngine

# Load defaults from iqrp/configs/risk/default.yaml
settings = RiskSettings.default()

# Or Hydra with overrides
settings = RiskSettings.from_hydra(
    overrides=[
        "var.method=fhs",
        "var.confidence=0.99",
        "sizing.kelly_fraction=0.25",
        "sizing.max_kelly=0.5",
        "limits.max_leverage=2.0",
    ]
)

engine = RiskIntelligenceEngine(settings)
```

Key blocks in `configs/risk/default.yaml`: `var`, `es`, `sizing`, `drawdown`, `limits`, `leverage`, `monte_carlo`, plus `seed`, `data_version`, `model_version`.

---

## Risk states

Canonical enum `iqrp.app.risk.RiskState`:

| State | Meaning |
|-------|---------|
| `NORMAL` | Within drawdown caution band |
| `CAUTION` | Elevated drawdown; tighten sizing |
| `REDUCED_RISK` | Material DD; reduce risk |
| `CAPITAL_PRESERVATION` | Severe DD; preserve capital |
| `TRADING_HALT` | Hard halt — `validate_position` rejects |

Transitions are deterministic from current drawdown vs configurable thresholds. See [DrawdownControl.md](DrawdownControl.md).

---

## RiskIntelligenceEngine API

```python
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings
import numpy as np

engine = RiskIntelligenceEngine(RiskSettings.default())
returns = np.array([...])  # point-in-time portfolio or asset returns
weights = np.array([0.4, 0.3, 0.3])

report = engine.calculate_risk(returns, weights=weights)
var = engine.var(returns, method="historical", confidence=0.95)
cvar = engine.cvar(returns, method="parametric", confidence=0.95)
es = engine.expected_shortfall(returns)
dd = engine.drawdown(returns)
state = engine.risk_state(returns)

decision = engine.validate_position(
    proposed_weight=0.08,
    weights=weights,
    returns=returns,
    realized_vol=0.15,
    participation=0.02,
    adv_coverage=50.0,
    forecast_confidence=0.9,
    asset_index=0,
)
assert decision.approved or "REJECTED" in decision.reason

size = engine.position_size(realized_vol=0.12, edge=0.02, confidence=0.8)
lev = engine.recommended_leverage(realized_vol=0.12, current_drawdown=0.04)
liq = engine.liquidity_risk(position_size=1e6, adv=5e6, spread=0.0005, price=100.0)
model = engine.model_risk_assessment(forecasts={"a": fa, "b": fb}, realizations=y)

engine.save("/tmp/risk_state.json")
engine2 = RiskIntelligenceEngine.load("/tmp/risk_state.json")
```

| Method | Role |
|--------|------|
| `calculate_risk` | Full `RiskReport` (tail, exposure, concentration, limits, state) |
| `portfolio_risk` | Portfolio vol from weights × covariance |
| `var` / `cvar` / `expected_shortfall` | Tail measures |
| `stress_test` / `reverse_stress` | Scenario analysis |
| `position_size` | Sized exposure with hard caps |
| `risk_contribution` | Marginal + component RC |
| `exposure` | Gross / net / long / short summary |
| `liquidity_risk` | ADV, participation, slippage, TTL |
| `drawdown` / `risk_state` | DD analytics + state |
| `check_limits` | Hierarchical limit evaluation |
| `validate_position` | Mandatory pre-trade gate (steps 1–9) |
| `recommended_leverage` | Dynamic leverage under hard clips |
| `model_risk_assessment` | Disagreement / uncertainty / drift |
| `monitor_snapshot` | Alerts + dashboard payload |
| `save` / `load` / `export_state` / `import_state` | Persistence |

---

## Auditability

Every `validate_position` decision appends to an internal audit log and returns `RiskDecision.audit` with:

- Timestamp (UTC ISO)
- Proposed weight / asset index
- Forecast confidence (informational — cannot override hard limits)
- `data_version` / `model_version`
- Limit, drawdown, and leverage parameters
- Drawdown snapshot
- Explicit `reason` string (`APPROVED` / `APPROVED_WITH_WARNINGS` / `REJECTED: ...`)

```python
decision = engine.validate_position(...)
print(decision.reason)
print(decision.audit["timestamp"], decision.audit["parameters"])
payload = engine.export_state()  # includes recent audit_log
```

---

## Pre-trade gate (validate_position)

Conceptual checklist enforced before execution — see [RiskLimits.md](RiskLimits.md) for detail:

1. Position validation  
2. Exposure validation  
3. Liquidity validation  
4. Concentration validation  
5. Portfolio risk validation  
6. Drawdown validation  
7. Leverage validation  
8. Model confidence validation (scales size only; never lifts hard limits)  
9. Global risk limits / `TRADING_HALT`  

---

## Integration hooks (import-only)

Risk **imports** sibling packages. It must not patch their internals or write into their stores. Contracts below are consumption patterns for composition roots (CLI, API, workers).

### Forecast Intelligence

```python
from iqrp.app.forecasting.intelligence import ForecastIntelligenceEngine

fi = ForecastIntelligenceEngine(...)
# Use point forecasts / confidence as *inputs* to sizing — never as limit overrides
confidence = float(fi_result.confidence)  # example field from FI payload
sizing = engine.position_size(realized_vol=rv, confidence=confidence)
assessment = engine.model_risk_assessment(
    forecasts={"m1": f1, "m2": f2},
    realizations=y_hat_aligned,
)
```

See [ForecastIntelligence.md](ForecastIntelligence.md).

### Volatility Forecasting

Obtain σ exclusively from the volatility engine — do not re-implement EWMA/GARCH in Risk. See [RiskIntegration.md](RiskIntegration.md) and [VolatilityForecasting.md](VolatilityForecasting.md).

```python
from iqrp.app.forecasting.volatility import create_volatility_model, VolatilityTrainer

model = create_volatility_model("garch")
model.fit(frame, target_column="returns")
sigma = float(model.conditional_volatility()[-1])
fc = model.forecast(frame, horizon=1)
# Pass sigma / forecast vol into engine.position_size / recommended_leverage / VaR scaling
```

### Regime Intelligence

```python
from iqrp.app.regimes import RegimeDetector  # or fitted RegimeModel

regimes = model.predict(frame)           # hard IDs, length T
probs = model.predict_proba(frame)       # soft (T, K)
# Map latest label to sizing / leverage regime string
regime_label = {0: "normal", 1: "high_vol", 2: "crisis"}.get(int(regimes[-1]), "transition")
engine.position_size(realized_vol=rv, regime=regime_label)
# Regime-conditioned MC: filter returns where regimes == k, then simulate
```

See [RegimeFramework.md](RegimeFramework.md), [MonteCarloRisk.md](MonteCarloRisk.md).

### Probability Engine

```python
from iqrp.app.math.probability import gaussian, get_distribution

# Use for research / custom parametric stress; live VaR/ES use risk.tail estimators
dist = gaussian(0.0, sigma)
```

See [ProbabilityEngine.md](ProbabilityEngine.md).

### Feature Platform

```python
from iqrp.app.features import get_feature, FeatureQueryService

# Liquidity / vol features as PIT inputs to liquidity_risk and sizing
adv = float(get_feature("adv_20d", frame)[-1])  # illustrative feature name
```

See [FeatureEngineering.md](FeatureEngineering.md), [FeaturePipeline.md](FeaturePipeline.md).

### Simulation Engine

```python
from iqrp.app.simulation import MarketSimulator, Scenario

sim = MarketSimulator()
market = sim.simulate(Scenario.from_settings(name="stress", model="merton_jump", n_steps=2000))
# Validate VaR / ES / limits on synthetic paths — never for live alpha
report = engine.calculate_risk(market.returns)
```

See [SimulationEngine.md](SimulationEngine.md).

### Portfolio / Execution interfaces

`iqrp.app.portfolio` and `iqrp.app.execution` are composition consumers:

```python
# Portfolio optimizer proposes weights → Risk validates → Execution may send
decision = engine.validate_position(
    proposed_weight=proposed_w_i,
    weights=current_weights,
    returns=pit_returns,
    realized_vol=sigma,
    participation=part,
    adv_coverage=adv_cov,
    forecast_confidence=conf,
)
if not decision.approved:
    raise RuntimeError(decision.reason)
# Only then: execution.submit(...)
```

Portfolio and execution packages must call Risk; Risk must not import concrete order routers or optimizers in the opposite direction for live decisions.

---

## Serialization

```python
path = engine.save("artifacts/risk/engine_state.json")
restored = RiskIntelligenceEngine.load(path)
```

Persisted payload includes settings, recent audit log, last report, and version stamps.
