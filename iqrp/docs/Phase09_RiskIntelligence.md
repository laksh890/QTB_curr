# Phase 09 — Risk Intelligence Completion

Machine-readable validation: [`Phase09_RiskIntelligence_Validation.json`](Phase09_RiskIntelligence_Validation.json)

## Completed components

| Component | Location |
|-----------|----------|
| Risk Framework | `iqrp.app.risk` |
| Position Sizing | `iqrp.app.risk.sizing` |
| Portfolio Risk | `iqrp.app.risk.portfolio` |
| VaR / CVaR / ES | `iqrp.app.risk.tail` |
| Stress / Scenarios | `iqrp.app.risk.stress` |
| Monte Carlo Risk | `iqrp.app.risk.simulation` |
| Correlation / Dependency | `iqrp.app.risk.market` |
| Kelly + Capital Allocation | `iqrp.app.risk.capital` |
| Dynamic Leverage | `iqrp.app.risk.leverage` |
| Risk Limits | `iqrp.app.risk.limits` |
| Risk Intelligence Ensemble | `iqrp.app.risk.ensemble` |

## Integration hooks (only existing-module change)

**File:** `iqrp/app/risk/__init__.py`

**Change:** Re-export `CapitalAllocator`, `CapitalSettings`, `CapitalAllocation`, `RiskBudget`, `RiskIntelligenceEnsemble`, `EnsembleSettings`, `RiskScore`, `RiskAssessment`, `EnsembleDecision`, `DecisionAction`.

**Reason:** Canonical Phase 09 entry points for Forecast → Risk → Portfolio/Execution consumers without requiring deep imports.

No existing VaR, sizing, portfolio, limits, or orchestrator logic was rewritten.

## Canonical usage

```python
from iqrp.app.risk import (
    RiskIntelligenceEngine,
    CapitalAllocator,
    RiskIntelligenceEnsemble,
    RiskState,
)

engine = RiskIntelligenceEngine()
allocator = CapitalAllocator()
ensemble = RiskIntelligenceEnsemble(risk_engine=engine)

allocation = allocator.allocate(["s1", "s2"], method="risk_parity", cov=cov, capital=1e6)
assessment = ensemble.aggregate({"volatility": 0.12, "var": 0.02, "cvar": 0.03, "drawdown": 0.04})
decision = ensemble.validate_position(
    proposed_weight=0.05, weights=[0.1, 0.1], returns=returns, forecast_confidence=0.9
)
```

## Config

- `iqrp/configs/risk/default.yaml`
- `iqrp/configs/risk/capital/default.yaml`
- `iqrp/configs/risk/ensemble/default.yaml`

## Validation

```bash
python -m iqrp.app.risk.phase09
# or
from iqrp.app.risk.phase09 import validate_phase09, write_phase09_report
```
