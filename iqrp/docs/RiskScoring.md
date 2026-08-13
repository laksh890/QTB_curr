# Risk Scoring

Identity-preserving multi-dimension risk scores for the Risk Intelligence Ensemble. Scores live in `[0, 1]` where **1 = maximum risk**.

Package: `iqrp.app.risk.ensemble`  
Modules: `scorer.py`, `normalizer.py`, `weighting.py`  
Types: `RiskScore`, `NormalizedMetric`

Related: [Risk Ensemble](RiskEnsemble.md) · [Risk State Machine](RiskStateMachine.md) · [Risk Decision](RiskDecision.md)

---

## Principle: not blind averaging

The ensemble **never** collapses heterogeneous raw metrics into a single naive mean. Instead:

1. Each observation is normalized to a risk unit while **preserving `original_value`**.
2. Related metrics feed a **named dimension** (max within the family — worst signal wins inside the dimension).
3. Dimensions keep their identity on `RiskScore` even when missing (conservative fill + residual weight).
4. `overall` is a **weighted synthesis** blended with the peak observed dimension — not an equal average of raw inputs.

```python
from iqrp.app.risk.ensemble import RiskIntelligenceEnsemble

ens = RiskIntelligenceEnsemble()
score = ens.score({
    "volatility": 0.25,
    "var": 0.05,
    "cvar": 0.07,
    "drawdown": 0.12,
    "liquidity_score": 0.6,
})
score.market, score.tail, score.overall
score.weights_applied
score.contributors
score.metadata["scoring_method"]  # identity_preserving_weighted_synthesis
```

---

## Dimension scores

`RISK_DIMENSIONS` / `RiskScore` fields:

| Dimension | Source metric keys (aliases) | Interpretation |
|-----------|------------------------------|----------------|
| `market` | `volatility`, `vol`, `realized_vol`, `garch_vol`, `gap_risk` | Market / vol / gap risk |
| `tail` | `var`, `cvar`, `expected_shortfall`, `es`, `var_historical`, `var_monte_carlo` | Tail loss risk |
| `liquidity` | `liquidity_score`, `liquidity`, `liquidity_model`, `liquidity_observed` | Illiquidity risk (inverted from liquidity score) |
| `concentration` | `concentration`, `hhi`, `herfindahl` | Name / sleeve concentration |
| `correlation` | `correlation`, `corr`, `avg_correlation`, `corr_normal`, `corr_stress` | Crowding / co-movement |
| `drawdown` | `drawdown`, `current_drawdown`, `max_drawdown`, `dd` | Underwater depth |
| `model` | `model_risk`, `model_disagreement`, `forecast_uncertainty` (+ disagreement fallback) | Model / forecast risk |
| `operational` | `operational`, `ops_risk`, `operational_risk` | Ops / process risk |
| `overall` | Weighted synthesis of the eight | Gate input to state machine |

Within a dimension, the scorer takes the **maximum** of available normalized aliases so a single severe estimator cannot be diluted by a milder sibling.

---

## Normalization references

Configured under `normalization` in `iqrp/configs/risk/ensemble/default.yaml` and `EnsembleSettings.normalization`.

Mapping: raw → `[0, 1]` risk via `(raw - zero) / (one - zero)`, clipped. With `invert=True`, higher raw means **lower** risk (liquidity scores).

| Metric key | Default `zero` | Default `one` | Invert |
|------------|----------------|---------------|--------|
| `volatility` | 0.0 | 0.50 | no |
| `var` | 0.0 | 0.10 | no |
| `cvar` | 0.0 | 0.15 | no |
| `expected_shortfall` | 0.0 | 0.15 | no |
| `drawdown` | 0.0 | `drawdown.trading_halt` (0.20) | no |
| `liquidity_score` | 1.0 | 0.0 | **yes** |
| `concentration` | 0.0 | 0.50 | no |
| `correlation` | 0.0 | 1.0 | no |
| `model_risk` | 0.0 | 1.0 | no |
| `operational` | 0.0 | 1.0 | no |
| `gap_risk` | 0.0 | 0.10 | no |

```python
from iqrp.app.risk.ensemble.types import NormalizedMetric

# Each NormalizedMetric retains:
#   original_value  — as observed
#   normalized_value — [0, 1] risk
#   reference — {zero, one, invert, …}
#   method / timestamp / model_version
```

---

## Overall synthesis (not a blind average)

After filling dimensions:

1. Resolve dimension weights (`static`, `risk_budget`, `regime`, …).
2. Missing dimensions receive a **conservative floor** (derived from observed means, clipped), **not** zero risk.
3. Weight mass from missing dims is mostly redistributed to observed dims; a small residual keeps identity in the report.
4. Contributors: `w_d × score_d`.
5. Overall: `0.65 × weighted_sum + 0.35 × peak_observed`, clipped to `[0, 1]`.

This prevents a hot critical dimension from being washed out by many mild dimensions, and prevents absent soft dimensions from dominating.

---

## Missing dimensions

| Case | Score behavior |
|------|----------------|
| Some dims missing | Conservative floor fill; residual weight kept; metadata lists `missing_dimensions` |
| All dims missing | Floor near `missing_penalty` (default ~0.85 path); never all-zero “safe” |
| Model missing but disagreement pairs present | Model score from `overall_disagreement` |

Missing **critical** metric keys are handled at the aggregate / decision layer (fallback state / REJECT) — see [Risk Ensemble](RiskEnsemble.md).

---

## Weights and contributors

```python
score.weights_applied   # normalized weights actually used
score.contributors      # w_d * score_d per dimension
score.metadata          # missing_dimensions, regime, scoring_method, …
```

Default static weights (sum ≈ 1.0):

| market | tail | liquidity | concentration | correlation | drawdown | model | operational |
|--------|------|-----------|---------------|-------------|----------|-------|-------------|
| 0.18 | 0.20 | 0.12 | 0.10 | 0.10 | 0.18 | 0.07 | 0.05 |

---

## Usage tips

- Prefer feeding **multiple estimators** per family; disagreement surfaces model risk instead of being averaged away.
- Do not pre-average VaR methods into one number before the ensemble — pass `var_historical` and `var_monte_carlo` separately when available.
- Treat `overall` as a **gate signal**, not a research KPI; inspect dimension scores and contributors for attribution.
- Forecast confidence is **not** a scoring dimension and cannot rewrite hard dimension scores.
