# Feature Research Validation Engine

Institutional feature validation and statistical research for IQRP.

**Purpose:** produce quantitative evidence that a feature contains useful predictive information. This module does **not** create trading signals, portfolios, or model predictions for production trading.

## Location

`iqrp/app/features/research/`

## Configuration

All thresholds and evaluation modes are Hydra-configurable:

- Defaults: `iqrp/configs/research/default.yaml`
- Load: `ResearchSettings.from_hydra()` or `ResearchSettings.from_hydra(overrides=["scoring.accept_score_threshold=65"])`

No hardcoded decision thresholds live in code paths that matter — they are read from `ResearchSettings`.

## Workflow

```python
import polars as pl
from iqrp.app.features.research import FeatureResearchValidator, ResearchSettings

settings = ResearchSettings.from_hydra()
validator = FeatureResearchValidator(settings)
result = validator.validate(feature_frame)  # OHLCV + feature columns

for score in result.accepted():
    print(score.feature, score.score, score.reason)
```

## Evidence produced

| Module | Evidence |
|--------|----------|
| `feature_statistics` | Mean/median/var/std/skew/kurtosis/entropy/missing/inf/zero/unique/distribution |
| `correlation` | Pearson/Spearman/Kendall/distance/MI/MIC/cross/rolling + clusters + network |
| `redundancy` | Duplicates, near-duplicates, VIF, linear dependence, rolling-window families |
| `predictive_power` | IC, Rank IC, MI, R², accuracy/precision/recall/F1/AUC vs future targets |
| `stability` | Rolling mean/var stability, rolling IC/MI/corr, decay, parameter drift |
| `drift` | PSI, KS, mean-shift z, concept IC ratio + alerts |
| `importance` | Permutation, SHAP (or linear SHAP-lite), LOO, drop-one, RFE, SFS |
| `validator` | 0–100 score, accept/weak/reject, rankings |
| `reports` / `visualization` | Markdown + JSON + SVG charts |

## Time-series evaluation

Validation **never shuffles**. Modes (config `predictive.evaluation_mode`):

- `walk_forward`
- `expanding`
- `rolling`
- `blocked` (purged block CV)

## Targets (research only)

Built from `close`:

- `future_return`
- `future_volatility`
- `future_drawdown`
- `future_direction`
- `future_regime`

## Outputs

Under `research.output_dir` (default `data/reports/feature_research/`):

- `feature_research_report.md`
- `feature_research_report.json`
- `charts/*.svg`

## Downstream contract

Markov / HMM / boosting / transformers / portfolio optimizers must consume **accepted** features from this engine (or an explicit research override), not ad-hoc indicator recomputation.
