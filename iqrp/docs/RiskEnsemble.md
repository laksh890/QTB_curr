# Risk Intelligence Ensemble

Unified multi-dimension risk gate coordinating normalization, scoring, disagreement, confidence, calibration, state transitions, and pre-trade decisions.

Package: `iqrp.app.risk.ensemble`  
Primary type: `RiskIntelligenceEnsemble`  
Hydra config: `iqrp/configs/risk/ensemble/default.yaml`

Related: [Risk Scoring](RiskScoring.md) · [Risk State Machine](RiskStateMachine.md) · [Risk Decision](RiskDecision.md) · [Risk Framework](RiskFramework.md)

---

## Architecture

```text
                    metrics bag (vol, VaR, CVaR, DD, …)
                              │
                              ▼
                     MetricNormalizer  ── preserves original_value
                              │
                              ▼
         DisagreementAnalyzer · WeightResolver · RiskScorer
                              │
                              ▼
                   RiskScore (8 dims + overall)
                              │
                              ▼
                  EnsembleStateMachine (hysteresis)
                              │
                              ▼
              RiskAssessment → build_decision → EnsembleDecision
                              │
              optional RiskIntelligenceEngine.validate_position
                              │
                              ▼
                     APPROVE | APPROVE_REDUCED | REJECT | HALT
```

The ensemble **does not reimplement** VaR / CVaR / drawdown / liquidity. It imports `iqrp.app.risk.*` and optionally wraps `RiskIntelligenceEngine`.

Architectural rules (enforced in code):

1. Do not blindly average metrics; preserve dimension identity.
2. Missing critical risk info → conservative fallback (never assume zero risk / auto-approve).
3. Forecast confidence / Kelly **must not** override hard limits.
4. State transitions are deterministic with hysteresis.
5. A single noisy metric must not trigger `TRADING_HALT` unless `hard_halt_on_single=true`.

---

## Quick start

```python
from iqrp.app.risk.ensemble import RiskIntelligenceEnsemble, EnsembleSettings
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings

engine = RiskIntelligenceEngine(RiskSettings.default())
ens = RiskIntelligenceEnsemble(
    EnsembleSettings.default(),
    risk_engine=engine,  # optional; hard rejects win
)

metrics = {
    "volatility": 0.22,
    "var": 0.04,
    "cvar": 0.06,
    "drawdown": 0.08,
    "liquidity_score": 0.7,
    "concentration": 0.15,
    "correlation": 0.45,
}

assessment = ens.aggregate(metrics)
decision = ens.decision(
    assessment=assessment,
    proposed_exposure=0.9,
    forecast_confidence=0.95,  # cannot override hard caps
)
```

---

## Inputs

Typical metric bag keys (aliases accepted by normalizer / scorer):

| Family | Keys |
|--------|------|
| Market | `volatility`, `vol`, `realized_vol`, `garch_vol`, `gap_risk` |
| Tail | `var`, `cvar`, `expected_shortfall`, `es`, `var_historical`, `var_monte_carlo` |
| Liquidity | `liquidity_score`, `liquidity`, `liquidity_model`, `liquidity_observed` |
| Concentration | `concentration`, `hhi`, `herfindahl` |
| Correlation | `correlation`, `corr`, `avg_correlation`, `corr_normal`, `corr_stress` |
| Drawdown | `drawdown`, `current_drawdown`, `max_drawdown`, `dd` |
| Model | `model_risk`, `model_disagreement`, `forecast_uncertainty` |
| Operational | `operational`, `ops_risk`, `operational_risk` |

**Critical keys** (default): `volatility`, `var`, `cvar`, `drawdown`. Missing any of these triggers conservative fallback (`missing_metrics_fallback_state` / `missing_metrics_fallback_action`).

---

## Normalization (preserve originals)

`MetricNormalizer` / `normalize_metrics` map each observation to `[0, 1]` **risk** (1 = maximum risk) while retaining the raw value on `NormalizedMetric.original_value`.

```python
from iqrp.app.risk.ensemble.types import NormalizedMetric

# NormalizedMetric fields:
# name, original_value, normalized_value, method, reference, timestamp, …
```

References come from `EnsembleSettings.normalization` (Hydra). Liquidity uses `invert=True` so high liquidity score → low risk. Drawdown’s high end aligns with `drawdown.trading_halt`. See [Risk Scoring](RiskScoring.md).

---

## Scoring dimensions

Eight identity-preserving dimensions plus `overall` synthesis — details in [Risk Scoring](RiskScoring.md):

`market` · `tail` · `liquidity` · `concentration` · `correlation` · `drawdown` · `model` · `operational`

---

## Weighting modes

`weighting_scheme` (`EnsembleSettings` / Hydra):

| Scheme | Behavior |
|--------|----------|
| `static` | Fixed `static_weights` (default) |
| `risk_budget` | Tilt toward dimensions consuming more score mass |
| `regime` | Stress/crisis up-weight tail, drawdown, correlation, liquidity, market |
| `dynamic` | Blend static with live dimension heat |
| `calibration` | Emphasize dimensions with poor calibration diagnostics |
| `stress` | Stress-oriented reweight |
| `user_defined` | `user_defined_weights` overlays |

Weights **scale contribution**; they never erase a dimension’s identity in the report.

---

## Disagreement

`DisagreementAnalyzer` compares configured estimator pairs (e.g. historical vs Monte Carlo VaR, GARCH vs realized vol, normal vs stress correlation). High disagreement elevates model risk and reduces confidence. Pair list and thresholds live under `disagreement` in Hydra.

```python
disc = ens.disagreement(metrics)
# disc["overall_disagreement"], per-pair deltas, n_pairs_available
```

---

## Confidence

`ConfidenceEstimator` blends base confidence with:

- disagreement penalty
- sample-size ramp (`sample_size_floor` → `sample_size_full`)
- missing-metric penalty
- hard clip to `[min_confidence, max_confidence]`

Confidence informs leverage **within** state caps only — never unlocks `TRADING_HALT` or raises hard exposure limits.

---

## Calibration

`CalibrationEngine` / `run_calibration` checks VaR / ES hit rates against configured alphas and tolerance bands. Calibration outcomes feed diagnostics and optional `calibration` weighting — they do not authorize limit breaches.

---

## Safe failure

| Condition | Behavior |
|-----------|----------|
| Missing critical metrics | Fallback state (default `CAPITAL_PRESERVATION`) and action (default `REJECT`) |
| Fallback would yield `APPROVE` | Guard rewrites to fallback action |
| Engine hard reject | Ensemble soft score cannot override → `REJECT` / `HALT` |
| Invalid / non-finite metrics | Dropped from normalization; treated as missing where critical |
| Empty metric bag | Conservative floor scores, never “all clear” |

---

## Public API surface

| Method | Role |
|--------|------|
| `aggregate` | Full `RiskAssessment` |
| `score` | `RiskScore` from metrics or assessment |
| `confidence` | Scalar or measure |
| `disagreement` | Pairwise / overall disagreement dict |
| `risk_state` | State-machine transition |
| `decision` | `EnsembleDecision` from assessment or metrics |
| `validate_position` | Pre-trade gate (see below) |
| `recommended_leverage` | Leverage within hard state caps |

Supporting modules: `diagnostics`, `evaluator`, `visualization`, `serializer`.

---

## Integration with `RiskIntelligenceEngine.validate_position`

When `risk_engine` is attached:

```python
decision = ens.validate_position(
    proposed_weight=0.08,
    weights=current_weights,
    returns=returns,
    metrics=metrics,              # optional enrichment bag
    forecast_confidence=0.9,
    realized_vol=0.18,
    participation=0.04,
    adv_coverage=12.0,
    asset_index=0,
)
```

Flow:

1. Build proposed portfolio weights; compute `proposed_exposure = Σ|w|`.
2. Enrich metrics from returns / engine (`volatility`, `var`, `cvar`, `drawdown`) — point-in-time only.
3. Detect missing critical keys; `aggregate` → assessment.
4. If engine present, call `risk_engine.validate_position(...)`.
5. **Engine hard reject wins** — ensemble soft score cannot override; reasons cite the engine.
6. Engine `TRADING_HALT` forces ensemble state to halt.
7. Build `EnsembleDecision`; re-apply state exposure / leverage caps.

Forecast confidence is logged and may scale leverage **inside** caps only. See [Risk Decision](RiskDecision.md).

---

## Configuration

```python
from iqrp.app.risk.ensemble import EnsembleSettings

settings = EnsembleSettings.from_hydra(
    overrides=[
        "hard_halt_on_single=false",
        "min_dimensions_for_halt=2",
        "weighting_scheme=regime",
        "missing_metrics_fallback_action=REJECT",
    ]
)
```

Key defaults: `hard_halt_on_single: false`, `min_dimensions_for_halt: 2`, critical keys as above, state caps shrinking exposure from `NORMAL` → `TRADING_HALT`.
