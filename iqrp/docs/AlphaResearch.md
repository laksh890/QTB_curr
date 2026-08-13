# Alpha Research

Institutional alpha research for IQRP: discover candidates, evaluate predictive power, validate under multiple-testing discipline, backtest with leakage controls, analyze decay / regimes / capacity, then promote or retire through an auditable registry.

**Package:** `iqrp.app.alpha`  
**Primary type:** `AlphaResearchEngine`  
**Hydra config:** `iqrp/configs/alpha/default.yaml`

Related: [SignalDiscovery](SignalDiscovery.md) · [SignalValidation](SignalValidation.md) · [MultipleTesting](MultipleTesting.md) · [BacktestValidation](BacktestValidation.md) · [SignalDecay](SignalDecay.md) · [SignalCapacity](SignalCapacity.md) · [SignalEnsemble](SignalEnsemble.md) · [SignalLifecycle](SignalLifecycle.md) · [Phase 11 summary](Phase11_AlphaResearch.md)

---

## Placement

```text
Features / Forecasts / Market data
              │
              ▼
     CandidateGenerator  ──► SignalRegistry (CANDIDATE)
              │
              ▼
      AlphaResearchEngine
   evaluate → validate → backtest
   decay / regimes / capacity
   compare → rank
              │
              ▼
   approve / degrade / retire
              │
              ▼
   APPROVED research alpha  ≠  trading permission
              │
              ▼
   Risk Intelligence → Portfolio → Execution
```

Alpha Research produces **research-approved** signals. It never grants trading authority. Risk Intelligence remains the gate for live risk and portfolio decisions.

---

## Architectural rules (14)

| # | Rule |
|---|------|
| 1 | **Statistical significance alone ≠ alpha.** A low p-value is a research lead, not an approval criterion by itself. |
| 2 | **Historical Sharpe alone cannot approve.** Gross or net Sharpe is diagnostic only; `allow_sharpe_only_approval` defaults to `false`. |
| 3 | **`economic_hypothesis` required for APPROVED.** Substantive rationale (≥ `min_hypothesis_chars`, default 20) explaining *why* the edge should exist. |
| 4 | **Alpha approval ≠ trading approval.** Risk Intelligence is not bypassed; approval extras record this explicitly. |
| 5 | **Point-in-time only.** Signal helpers use past windows; forward returns are *labels*, never features. |
| 6 | **Rejected experiments are preserved.** Registry never silently deletes rejects; `preserve_rejected: true`. |
| 7 | **Discovery emits candidates, not approved alpha.** Templates set `claims_profitability=False`. |
| 8 | **Auditable transitions.** Every status change records from→to, reason, actor, timestamp. |
| 9 | **Evaluate + validate evidence before APPROVED.** Engine refuses promotion without attached validation diagnostics. |
| 10 | **Trial budget must be tracked.** Multiple-testing adjustments and `ExperimentTracker` account for search intensity. |
| 11 | **Capacity and costs matter.** Deployable AUM, ADV participation, and impact enter ranking and retirement — not just IC. |
| 12 | **Ensemble weights are not Sharpe-only.** Composite weighting caps Sharpe contribution; IC, stability, capacity, decay, correlation dominate. |
| 13 | **Leakage-safe validation.** Walk-forward, purged CV, embargo, and nested CV keep train/test temporally honest. |
| 14 | **Terminal statuses are final for promotion.** `REJECTED` and `RETIRED` do not re-enter the approval path without a new experiment. |

---

## Quick start

```python
import numpy as np
from iqrp.app.alpha import (
    AlphaResearchEngine,
    AlphaSettings,
    SignalDefinition,
    SignalStatus,
)

rng = np.random.default_rng(42)
returns = rng.normal(0, 0.01, 500)
momentum = np.concatenate([[0.0], np.cumsum(returns[:-1])])  # PIT proxy

eng = AlphaResearchEngine(AlphaSettings.default())

# 1) Discover research candidates (NOT alpha)
candidates = eng.discover(returns=returns, features={"mom": momentum})

# 2) Register an explicit definition
defn = SignalDefinition(
    name="mom_20",
    version="0.1.0",
    formula="cumsum(returns).shift(1)",
    features=("returns",),
    lookback=20,
    horizon=1,
    universe="liquid_us",
    frequency="1d",
    direction="long_short",
    expected_relationship="positive",
    economic_hypothesis=(
        "Underreaction to gradual information flow produces short-horizon "
        "continuation in liquid names; edge should decay as horizon lengthens."
    ),
    owner="research",
    signal_type="momentum",
)
rec = eng.register(defn, signal=momentum)

# 3) Evaluate + validate (attach evidence to experiment)
eng.evaluate(momentum, returns, experiment_id=rec.experiment_id, definition=defn)
eng.validate(momentum, returns, experiment_id=rec.experiment_id, n_trials=25)

# 4) Economics / decay / stress
decay = eng.analyze_decay(momentum, returns)
cap = eng.analyze_capacity(turnover=0.2, adv=5e7, max_participation=0.05)
stress = eng.stress_test(momentum, returns)

# 5) Research approval (still not trading approval)
approved = eng.approve(
    rec.experiment_id,
    reason="IC significant after BH-FDR; DSR/PBO acceptable; hypothesis documented",
    actor="lead_researcher",
)
assert approved.status == SignalStatus.APPROVED
```

---

## `AlphaResearchEngine` pipeline

| Method | Role | Notes |
|--------|------|-------|
| `discover(...)` | Multi-source candidate generation | Features, forecasts, TS templates, statistical screens, symbolic formulas |
| `register(definition, signal=...)` | Persist experiment as `CANDIDATE` (default) | Always tracks `economic_hypothesis` on the definition |
| `evaluate(signal, forward_returns, ...)` | IC / Rank IC / stability / hit-rate report | Attach via `experiment_id` |
| `validate(signal, forward_returns, n_trials=...)` | Significance, bootstrap, permutation, MT, DSR, PBO | Writes `diagnostics["validation"]` |
| `backtest(signal, returns, ...)` | Causal long/short PnL with optional costs | Sharpe fields are diagnostic only |
| `stress_test(signal, returns, regimes=...)` | Vol shocks, sign flip, optional regime block | Does not approve |
| `analyze_decay(signal, returns, horizons=...)` | Multi-horizon IC, half-life, optimal hold | See [SignalDecay](SignalDecay.md) |
| `analyze_regimes(signal, returns, regimes)` | Conditional IC / performance by regime | |
| `analyze_capacity(turnover=..., adv=..., **kw)` | ADV participation capacity | See [SignalCapacity](SignalCapacity.md) |
| `compare(signals, returns=...)` | Correlation + redundancy (+ optional IC) | Triage only |
| `rank(candidates)` | Multi-factor research ranking | Not Sharpe-only |
| `approve(experiment_id, reason=..., actor=...)` | Promote through lifecycle to `APPROVED` | Hard gates; RI not bypassed |
| `degrade(experiment_id, reason=...)` | `PROVISIONAL`/`APPROVED` → `DEGRADED` (or reject pre-approval) | |
| `retire(experiment_id, reason=...)` | Terminal retirement with audit trail | Cannot retire `REJECTED` |
| `research_report(experiment_id)` | Fetch attached report or governance warnings report | |
| `save` / `load` | Persist engine state, definitions, reports | |

Constructor:

```python
from iqrp.app.alpha import AlphaResearchEngine, AlphaSettings
from iqrp.app.alpha.base.signal_registry import SignalRegistry

eng = AlphaResearchEngine(
    settings=AlphaSettings.default(),
    registry=SignalRegistry(),  # or get_default_registry()
)
```

### Approve gates (enforced)

`approve()` raises `ApprovalError` when any of the following fail:

1. Missing or thin `economic_hypothesis` (when `require_hypothesis=True`).
2. Sharpe-only approval reason / evidence while `scoring.allow_sharpe_only_approval` is `false`.
3. Missing evaluate/validate evidence keys on the experiment report (`significance`, `bootstrap`, `permutation`, `deflated_sharpe`, `pbo`, `validate`, etc.).

On success, extras include:

```python
{
    "risk_intelligence_not_bypassed": True,
    "alpha_approval_is_not_trading_approval": True,
}
```

---

## `SignalDefinition` fields

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | Non-empty signal name |
| `version` | `str` | Semver-style research version; `definition_id = name@version` |
| `formula` | `str` | Human/machine-readable expression of the transform |
| `features` | `tuple[str, ...]` | Input feature names |
| `lookback` | `int` (≥1) | Past window used by the signal |
| `horizon` | `int` (≥1) | Intended prediction horizon (bars) |
| `universe` | `str` | Research universe tag |
| `frequency` | `str` | Sampling frequency (e.g. `1d`) |
| `direction` | `long_short` \| `long_only` \| `short_only` \| `neutral` | Intended book sidedness |
| `expected_relationship` | `positive` \| `negative` \| `nonmonotonic` \| `unknown` | Prior on sign |
| `economic_hypothesis` | `str` | **Why** the relationship should exist (mandatory for APPROVED) |
| `owner` | `str` | Research owner |
| `signal_type` | momentum / mean_reversion / trend / volatility / volume / cross_sectional / event / alternative / statistical / symbolic / custom | Taxonomy |
| `parameters` | `dict` | Free-form hyperparameters |
| `tags` | `tuple[str, ...]` | Search / filtering tags |
| `created_at` | `datetime` | UTC creation time |
| `notes` | `str` | Free-form research notes |

```python
from iqrp.app.alpha import SignalDefinition

d = SignalDefinition(
    name="cs_value_resid",
    version="1.0.0",
    formula="residualize(book_to_price, sector)",
    features=("book_to_price", "sector"),
    lookback=1,
    horizon=5,
    universe="russell_3000",
    frequency="1d",
    direction="long_short",
    expected_relationship="positive",
    economic_hypothesis=(
        "Within-sector valuation spreads compensate for slow capital "
        "reallocation; residualization removes industry risk premia."
    ),
    owner="quant_research",
    signal_type="cross_sectional",
    parameters={"neutralize": "sector"},
    tags=("value", "neutralized"),
)
assert d.definition_id == "cs_value_resid@1.0.0"
```

Empty hypothesis is allowed at `CANDIDATE` construction so discovery can register leads; promotion to `APPROVED` still requires a substantive hypothesis.

---

## Economic hypothesis requirement

An economic hypothesis is a falsifiable mechanism, not a restatement of in-sample IC:

- **Good:** “Post-earnings drift from gradual information diffusion; expected Rank IC positive at 5–20d, decaying thereafter.”
- **Bad:** “IC was 0.08 on 2019–2024.”

Scoring allocates `weight_economic_hypothesis` (default 0.20) in composite research scores. Registry and engine both refuse thin hypotheses at APPROVED.

---

## Research approval vs trading approval

| Layer | Meaning |
|-------|---------|
| Research `APPROVED` | Signal cleared alpha-research governance (hypothesis, validation evidence, non-Sharpe-only gates). |
| Trading permission | Separate path through Risk Intelligence limits, portfolio construction, and execution capacity. |

`AlphaResearchEngine.approve` never calls Risk Intelligence APIs to grant size or bypass limits. Downstream systems must re-validate.

---

## Hydra config — `configs/alpha/default.yaml`

Load via `AlphaSettings.from_hydra()` or OmegaConf.

| Section | Key defaults | Role |
|---------|--------------|------|
| root | `seed=42`, `owner_default=research`, `universe_default=default`, `frequency_default=1d` | Reproducibility / defaults |
| `discovery` | momentum/MR lookbacks, `statistical_min_abs_ic=0.02`, `auto_register=true` | Candidate generation |
| `research` | `horizons=[1,2,5,10]`, `stability_window=60` | Evaluation / decay grid |
| `scoring` | predictive/stability/persistence/hypothesis weights; `require_economic_hypothesis=true`; `allow_sharpe_only_approval=false` | Promotion gates |
| `governance` | `preserve_rejected=true`, `auditable_transitions=true`, `terminal_statuses=[REJECTED, RETIRED]` | Registry policy |

```python
from iqrp.app.alpha import AlphaSettings, AlphaResearchEngine

settings = AlphaSettings.from_hydra(
    overrides=["scoring.min_hypothesis_chars=40", "discovery.auto_register=false"]
)
eng = AlphaResearchEngine(settings)
```

---

## Experiment registry

`SignalRegistry` / `get_default_registry()` stores every trial, including rejects:

```python
from iqrp.app.alpha import get_default_registry, SignalStatus

reg = get_default_registry()
rejects = reg.rejected_experiments()  # preserved for audit / learning
trail = reg.audit_trail(experiment_id)
```

Rejected experiments remain listable (`include_rejected=True` by default) so false-discovery accounting and institutional learning are possible. See [SignalLifecycle](SignalLifecycle.md) and [ExperimentRegistry](ExperimentRegistry.md).

---

## Package exports

Canonical imports:

```python
from iqrp.app.alpha import (
    AlphaResearchEngine,
    AlphaSettings,
    AlphaSerializer,
    AlphaSignal,
    ApprovalError,
    ExperimentRecord,
    SignalDefinition,
    SignalRegistry,
    SignalResearchReport,
    SignalStatus,
    get_default_registry,
    validate_phase11,
    write_phase11_report,
)
```

---

## Validation

```bash
python -m iqrp.app.alpha.phase11
```

```python
from iqrp.app.alpha import validate_phase11, write_phase11_report

report = validate_phase11()
path = write_phase11_report()  # writes Phase11_AlphaResearch_Validation.json
```

Machine-readable report: [`Phase11_AlphaResearch_Validation.json`](Phase11_AlphaResearch_Validation.json).
