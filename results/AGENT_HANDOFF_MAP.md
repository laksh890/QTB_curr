# IQRP Research Handoff Map (Prompts 35–40)

**Purpose:** Give a new agent (or human) a single entry point: what to read first, what is authoritative, what is immutable, and what the next authorized phase may use.

**Last research status:** Prompt 40 `CONSOLIDATION_COMPLETE_RESEARCH_SET`  
**Proven profitability:** NO  
**Production / live ready:** NO  

---

## Start here (reading order)

| Priority | File | Why |
|----------|------|-----|
| **1** | `results/candidate_consolidation/final_report.md` | **Current frontier.** Reduced research set after consolidation. |
| **2** | `results/candidate_consolidation/final_candidate_set.json` | The **8** `DISTINCT_RESEARCH_CANDIDATES` + **10** ensemble IDs to carry forward. |
| **3** | `results/candidate_consolidation/consolidation_config.json` | Frozen thresholds, weighting formulas, selection rules (no OOS tuning). |
| **4** | `results/model_driven_alpha_campaign/campaign_report.md` | Prompt 39 model→alpha campaign outcome (129 CANDIDATEs from 4806 exps). |
| **5** | `results/final_system_architecture_audit/` | Architecture complete for research/sim; not production/live. |
| **6** | `results/alpha_research_btc_full_audit/` | Prompt 36 validity: **CONDITIONALLY_VALID**, statistical validity **LIMITED**. |
| **7** | `results/alpha_research_btc_full/final_report.md` | Prompt 35 reference-signal campaign (0 CANDIDATE). Historical baseline only. |

**Do not start from Prompt 35/39 candidate rankings alone** if continuing research — they are upstream. **Prompt 40 final set is the working universe.**

---

## Authoritative “weights” / configs for next work

Use these as frozen inputs unless a later prompt explicitly authorizes changes:

| Asset | Path | Role |
|-------|------|------|
| Distinct research candidates | `results/candidate_consolidation/final_candidate_set.json` → `DISTINCT_RESEARCH_CANDIDATES` | Primary carry-forward set (8) |
| Ensemble research candidates | same file → `ENSEMBLE_CANDIDATES` | Secondary (10); not independent discoveries |
| Consolidation thresholds | `results/candidate_consolidation/consolidation_config.json` | Corr/redundancy/cluster/turnover/weight formulas |
| P39 campaign protocol | `results/model_driven_alpha_campaign/campaign.json` | Datasets, MAX_BARS, horizons, costs, model specs |
| Cost scenarios | `campaign.json` → `cost_scenario_defs` (BASE/MODERATE/ADVERSE) | Same accounting everywhere |
| Dataset registry | `dataset_registry.json` | BTCUSDT registered datasets `@1.0.0` (history ends 2024-12-31) |
| Ensemble registry | `results/candidate_consolidation/ensemble_registry.json` | Methods + member IDs + validation-only weights |

**Selection note:** Prompt 40 representatives were chosen with **validation** Sharpe + independence/turnover rules. OOS was evaluation-only. Do not re-rank by OOS Sharpe unless a new protocol says so.

---

## Immutable vs writable

| Path | Rule |
|------|------|
| `results/alpha_research_btc_full/` | **IMMUTABLE** (Prompt 35) |
| `results/alpha_research_btc_full_audit/` | **IMMUTABLE** (Prompt 36) |
| `results/model_driven_alpha_campaign/` | **IMMUTABLE** (Prompt 39) |
| `results/candidate_consolidation/` | Latest consolidation; treat as **source of truth** until a later prompt supersedes |
| `results/model_alpha_integration/` | Prompt 37 adapter validation (wiring only) |
| `results/unified_trading_pipeline/` | Prompt 38 Alpha→Risk→Portfolio→Execution research path |

---

## Artifact maps by campaign

### A. Prompt 40 — Candidate consolidation (CURRENT)

Directory: `results/candidate_consolidation/`

| File | Contents |
|------|----------|
| `final_report.md` / `.json` | Status + 20 required answers |
| `final_candidate_set.json` | Reduced universe (8 + 10 ensembles) |
| `consolidation_config.json` | Protocol / thresholds |
| `candidate_clusters.json` | 8 behavioral clusters + representatives |
| `candidate_cluster_summary.json` | Size histogram |
| `candidate_dependency_matrix.json` / `.csv` | Pairwise dependence |
| `redundancy_analysis.json` | DISTINCT / RELATED / HIGHLY_REDUNDANT |
| `model_diversification.json` | Model-family dependence |
| `timeframe_diversification.json` | TF dependence |
| `horizon_diversification.json` | Holding-horizon clones |
| `direction_diversification.json` | LONG vs SHORT |
| `drawdown_dependence.json` | Simultaneous loss / DD overlap |
| `cost_dependence.json` | BASE→ADVERSE sensitivity |
| `ensemble_registry.json` | Predeclared ensembles A–D |
| `ensemble_results.json` | Metrics by cost scenario |
| `ensemble_comparison.json` | Gate survivors |
| `rejection_summary.json` | What was dropped and why |
| `reproducibility_report.json` | Deterministic rerun + defect fix note |
| `candidate_universe.json` | All 129 with reconstructed metrics |

Code: `iqrp/app/backtesting/alpha_research/consolidation/`  
Run: `python -m iqrp.app.backtesting.alpha_research.consolidation`

### B. Prompt 39 — Model-driven alpha campaign

Directory: `results/model_driven_alpha_campaign/`

| File | Contents |
|------|----------|
| `campaign_report.md` / `.json` | Campaign status + answers |
| `campaign.json` | Frozen protocol |
| `experiment_registry.json` | All experiments (incl. 129 BASE CANDIDATE) |
| `candidate_rankings.json` | Top rankings (cap; use registry for full 129) |
| `model_family_summary.json` | Per-family gate counts |
| `timeframe_summary.json` / `horizon_summary.json` | TF × horizon grid |
| `cost_summary.json` / `trade_frequency_summary.json` | Costs & trading intensity |
| `multiple_testing.json` | FDR on BASE LONG_SHORT |
| `model_fit_log.json` / `unavailable_log.json` | Fits & UNAVAILABLE reasons |

Code: `iqrp/app/backtesting/alpha_research/model_campaign/`  
Run: `python -m iqrp.app.backtesting.alpha_research.model_campaign`

### C. Prompt 37–38 — Wiring (not alpha discovery)

| Dir | Role |
|-----|------|
| `results/model_alpha_integration/` | Model → adapter → SignalRegistry → Alpha path validation |
| `results/unified_trading_pipeline/` | Candidate → Risk → Portfolio constraints → Execution sim |

Code:  
- `iqrp/app/backtesting/alpha_research/adapters/`  
- `iqrp/app/backtesting/unified_pipeline/`

### D. Prompt 35–36 — Reference baseline + validity

| Dir | Role |
|-----|------|
| `results/alpha_research_btc_full/` | Reference signals only; **0 CANDIDATE** |
| `results/alpha_research_btc_full_audit/` | Validity audit; statistical validity LIMITED |

### E. Architecture audit

| Dir | Role |
|-----|------|
| `results/final_system_architecture_audit/` | RESEARCH_READY / PAPER-SIM yes; PRODUCTION/LIVE no |
| `results/architecture_implementation_audit/` | Models existed but were unwired before P37 |

---

## Claim ladder (never collapse)

```
MODEL IMPLEMENTED
≠ FORECAST
≠ SIGNAL
≠ BACKTESTABLE
≠ SURVIVES OOS
≠ SURVIVES COSTS
≠ ROBUST
≠ DISTINCT AFTER CONSOLIDATION
≠ PROFITABLE STRATEGY
≠ PRODUCTION / LIVE READY
```

Current position: **distinct research candidates + ensembles exist; profitability not proven; live not authorized.**

---

## Data constraints

- Symbol: **BTCUSDT**
- Datasets: `btcusdt_intraday_{1m,5m,15m,30m,1h}@1.0.0` via `dataset_registry.json`
- Available history ends **2024-12-31** (no fabricated 2025/2026)
- Prompt 39 used `MAX_BARS` subsample (see `campaign.json`)

---

## Code map (for implementation follow-ons)

| Concern | Location |
|---------|----------|
| Reference signals | `iqrp/app/backtesting/alpha_research/signals.py`, `reference_signals.py` |
| Alpha engine / gates | `iqrp/app/backtesting/alpha_research/engine.py`, `types.py` |
| Model→signal adapters | `iqrp/app/backtesting/alpha_research/adapters/` |
| P39 campaign | `iqrp/app/backtesting/alpha_research/model_campaign/` |
| P40 consolidation | `iqrp/app/backtesting/alpha_research/consolidation/` |
| Unified Risk/Portfolio/Exec | `iqrp/app/backtesting/unified_pipeline/` |
| Tests | `iqrp/tests/unit/backtesting/test_model_driven_campaign.py`, `test_candidate_consolidation.py`, `test_model_alpha_adapters.py`, `test_unified_trading_pipeline.py` |

Env: prefer `.venv/bin/python`.

---

## What a next agent should / should not do

**Should (only if a new prompt authorizes):**

- Load `final_candidate_set.json` as the research universe
- Reuse P39/P40 protocols, costs, datasets, adapters
- Keep claim distinctions and immutability of P35–P39 dirs

**Should not (until explicitly authorized):**

- Paper trading / broker / live
- Portfolio-weight optimization against OOS
- New models or datasets
- Architecture redesign
- Overwriting Prompt 35/36/39 artifacts
- Claiming profitability from research Sharpes

---

## One-line pointer for prompts

> **Resume from:** `results/candidate_consolidation/final_report.md` + `final_candidate_set.json`.  
> **Context map:** this file (`results/AGENT_HANDOFF_MAP.md`).  
> **Upstream campaign:** `results/model_driven_alpha_campaign/campaign_report.md` (immutable).
