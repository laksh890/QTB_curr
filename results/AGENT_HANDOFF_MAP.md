# IQRP Research Handoff Map (Prompts 35–43)

**Purpose:** Short entry point for a new agent. For the **full architectural roadmap, feature inventory, prompt history, and next-phase plan**, read:

> **`results/ARCHITECTURAL_ROADMAP_AND_HANDOVER.md`** (authoritative long-form handover)

**Last research status:** Prompt 43 paper sim = `PAPER_TRADING_CANDIDATE` (sequential assumed-microstructure execution on frozen A/B/C). Prior: frozen 2024→2025 holdout `PROVEN_RESEARCH_PROFITABILITY` for 2 MTF evidence candidates.  
**LIVE_READY:** NO  
**Broker / live orders:** STOP — not authorized.

---

## Start here (reading order)

| Priority | File | Why |
|----------|------|-----|
| **0** | `results/ARCHITECTURAL_ROADMAP_AND_HANDOVER.md` | **Complete architecture + progress + roadmap** for agent handover. |
| **1** | `results/paper_trading_validation/final_report.md` | **Current frontier.** Sequential paper sim + realistic assumed fills. |
| **2** | `results/paper_trading_validation/paper_trading_status.json` | Gates, status, reproducibility. |
| **3** | `results/frozen_2024_2025_holdout/final_report.md` | Research freeze + 2025 independent holdout evidence. |
| **4** | `results/frozen_2024_2025_holdout/firewall_audit.json` | Hard temporal firewall evidence. |

---

## Prompt 43 — Paper trading & realistic execution (CURRENT)

Directory: `results/paper_trading_validation/`

| File | Role |
|------|------|
| `final_report.md` / `.json` | Status + required answers |
| `candidate_results.json` | A/B/C + combos under BASE assumed costs |
| `execution_results.json` / `fill_analysis.json` | Order/fill samples (ASSUMED_OHLCV_MICROSTRUCTURE) |
| `cost_analysis.json` | BASE / MODERATE / ADVERSE sensitivity |
| `failure_injection.json` | Operational failure + kill-switch tests |
| `position_reconciliation.json` | Zero unexplained drift |
| `portfolio_comparison.json` | Combo diversification |
| `risk_events.json` / `paper_trading_status.json` / `test_summary.json` | Risk + status + tests |

Code: `iqrp/app/paper_trading/`  
Run: `.venv/bin/python -m iqrp.app.paper_trading` (add `--smoke` for short path)

### Frozen sleeves (immutable; no 2025 retune)

| Label | Candidate |
|-------|-----------|
| A | `mdc_99aa952c5d5f6ff7` |
| B | `mdc_6f008c954ea26bf5` |
| C | `mdc_678609c534d68189` |

**Status:** `PAPER_TRADING_CANDIDATE` — **not** `PROVEN PROFITABLE`, **not** `LIVE_READY`. Costs are assumed (no observed bid/ask). ADVERSE scenario can erase A’s paper P&L.

---

## Frozen 2024 → Independent 2025 holdout

Directory: `results/frozen_2024_2025_holdout/`

| File | Role |
|------|------|
| `final_report.md` / `.json` | Status + required answers |
| `firewall_audit.json` | Research ≤2024 / holdout=2025 hard split |
| `dataset_provenance.json` / `dataset_quality.json` | Immutable 2025 slice (complete 525,600×1m) |
| `frozen_candidate_manifest.json` | P40/P39 definition checksums |
| `holdout_results.json` / `decision_matrix.json` | Full-year 2025 metrics + gates |
| `sharpe_independent_recalculation.json` | Engine vs independent Sharpe |
| `quarterly_analysis.json` / `cost_analysis.json` / `statistical_validation.json` | Diagnostics |
| `portfolio_comparison.json` | Pre-2025 weights → 2025 eval |
| `reproducibility_report.json` | Dual-run PASS |

Code: `iqrp/app/backtesting/frozen_2025_holdout/`  
Run: `.venv/bin/python -m iqrp.app.backtesting.frozen_2025_holdout`

### Evidence focus (2025)

| Candidate | Status | 2025 net Sharpe | ADVERSE |
|-----------|--------|----------------:|---------|
| `mdc_99aa952c5d5f6ff7` | PROVEN_RESEARCH_PROFITABILITY | ~7.03 | survives |
| `mdc_6f008c954ea26bf5` | PROVEN_RESEARCH_PROFITABILITY | ~5.37 | survives |
| `mdc_678609c534d68189` | PAPER_TRADING_CANDIDATE | ~6.23 | fails |

**LIVE_READY:** NO. ML controls (CatBoost/XGB combo) largely **REJECTED** on 2025.

---

## Independent candidate validation

Directory: `results/independent_candidate_validation/`

| File | Role |
|------|------|
| `final_report.md` / `.json` | **INVALID_HOLDOUT** + 11 required answers |
| `data_provenance.json` | Temporal firewall; 1-day local Vision remnant; network FAIL |
| `holdout_results.json` | Frozen primary + negative control diagnostics |
| `walk_forward_results.json` / `regime_results.json` / `cost_results.json` | Diagnostics only |
| `statistical_results.json` / `bootstrap_results.json` | Dependence-aware; insufficient sample |
| `capacity_results.json` | `ESTIMATE_ONLY` (OHLCV) |
| `negative_control_results.json` | `mdc_6f008c954ea26bf5` |
| `reproducibility_report.json` / `test_summary.json` | Dual-run + pytest |

Code: `iqrp/app/backtesting/independent_validation/`  
Run: `.venv/bin/python -m iqrp.app.backtesting.independent_validation`

**Verdict:** Only **1** calendar day after P42 firewall is available free/local; Binance network acquisition failed. Protocol ⇒ **INVALID_HOLDOUT**. Short-window Sharpes are **not** replication. **NO_PAPER_TRADING_CANDIDATE**. **LIVE_READY: NO**. Paid L2/tick → **STOP_BEFORE_PURCHASE**.

---

## Prompt final holdout — Independent frozen-alpha holdout

Directory: `results/final_holdout_validation/`

| File | Contents |
|------|----------|
| `final_report.md` / `.json` | Status + 19 required answers |
| `data_provenance.json` / `.md` | Registered vs holdout independence |
| `candidate_freeze.json` | Exact P39/P42 definition checksums |
| `causality_audit.json` | Next-bar / lag checks |
| `holdout_results.json` | Per-candidate holdout metrics |
| `cost_stress.json` | BASE / MODERATE / ADVERSE |
| `regime_results.json` | Causal regime slices |
| `statistical_validation.json` | n_eff / HAC — generally INSUFFICIENT (1 day) |
| `degradation_analysis.json` | vs Prompt 42 |
| `reconciliation.json` / `reproducibility.json` | Dual-run + cascade recon |

Code: `iqrp/app/backtesting/final_holdout/`  
Run: `.venv/bin/python -m iqrp.app.backtesting.final_holdout`

### Holdout outcome (do not retune)

| Candidate | Holdout class | Notes |
|-----------|---------------|-------|
| `mdc_99aa952c5d5f6ff7` | WEAK_EVIDENCE | Positive BASE/MODERATE; sample too short |
| `mdc_678609c534d68189` | WEAK_EVIDENCE | Positive BASE/MODERATE; sample too short |
| `mdc_6f008c954ea26bf5` | REJECTED | Failed holdout replication (LONG 15m) |

**PAPER_TRADING_CANDIDATE:** none  
**LIVE_READY:** NO  

Holdout source: Binance Vision July 2026 ZIP bars after registered `@1.0.1` truncation at `2026-07-31 00:00:00` (~1 calendar day). Not manufactured from pre-P42 history.

---

## Prompt 42 — Final trading validation (immutable)

Directory: `results/final_trading_validation/`

| File | Contents |
|------|----------|
| `final_report.md` / `.json` | Status + decision matrix + required answers |
| `profitability_gate.json` | Frozen gate outcomes |
| `candidate_results.json` | Per-candidate deep validation |
| `trading_behavior.json` | Trades/day, behavior class, long/short |
| `cost_analysis.json` | BASE / MODERATE / ADVERSE |
| `execution_realism.json` | Next-bar / no lookahead notes |
| `regime_analysis.json` | Causal vol/trend regime survival |
| `walk_forward_results.json` | Rolling folds |
| `statistical_validation.json` | n_eff, HAC/Newey-West, overlap adj. |
| `portfolio_comparison.json` | MV / RP / BL / HRP / constraints vs equal |
| `data_provenance.json` | Free-source survey + `@1.0.1` extended BTC |
| `experiment_registry.json` | Validation experiment IDs |
| `reproducibility_report.json` | Deterministic rerun |
| `test_summary.json` | Pytest evidence |
| `validation_config.json` | Frozen gates/grids |

Code: `iqrp/app/backtesting/final_validation/`  
Run: `.venv/bin/python -m iqrp.app.backtesting.final_validation`  
Smoke: `.venv/bin/python -m iqrp.app.backtesting.final_validation --smoke`

### Outcome summary (do not re-rank by OOS)

| Status | Count | Notes |
|--------|------:|-------|
| `PROFITABILITY_EVIDENCE` | 3 | MTF momentum family (15m L/S, 15m LONG, 5m SHORT) |
| `RESEARCH_ONLY` | 1 | 5m L/S — fails non-catastrophic ADVERSE |
| `FRAGILE` / `OOS_FAILED` | 4 | Ensemble, combo, CatBoost variants |

**Best diagnostic family:** MTF (multi-timeframe momentum)  
**Best diagnostic TF / holding:** 5m / 2 bars (diagnostic medians only — not selection)  
**Data:** Binance Vision `BTCUSDT` `@1.0.1` (2019-01 → 2026-07), **RESEARCH_GRADE** (OHLCV-only; no bid/ask/depth). Paid tick/L2 identified → **STOP_BEFORE_PURCHASE**.

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
≠ PROFITABILITY_EVIDENCE   ← Prompt 42 (3 candidates)
≠ HOLDOUT REPLICATED       ← final holdout (weak / partial)
≠ PROVEN PROFITABILITY (absolute)
≠ PAPER-READY / PAPER_TRADING_CANDIDATE
≠ LIVE_READY
```

---

## Authoritative frozen inputs

| Asset | Path | Role |
|-------|------|------|
| Distinct research candidates | `results/candidate_consolidation/final_candidate_set.json` | P42 validation universe (8) |
| P42 gate protocol | `iqrp/app/backtesting/final_validation/protocol.py` | Do not relax after results |
| Extended datasets | `dataset_registry.json` → `btcusdt_intraday_*@1.0.1` | Prefer over `@1.0.0` |
| Cost scenarios | `iqrp.app.backtesting.alpha_research.types.COST_SCENARIOS` | BASE/MODERATE/ADVERSE |

**Note:** `@1.0.0` and `@1.0.1` share parquet paths (in-place extension). Treat `@1.0.1` checksums as current; do not silently fill gaps.

---

## Immutable vs writable

| Path | Rule |
|------|------|
| `results/alpha_research_btc_full/` | **IMMUTABLE** (Prompt 35) |
| `results/alpha_research_btc_full_audit/` | **IMMUTABLE** (Prompt 36) |
| `results/model_driven_alpha_campaign/` | **IMMUTABLE** (Prompt 39) |
| `results/candidate_consolidation/` | **IMMUTABLE** input to P42 |
| `results/portfolio_construction_integration/` | **IMMUTABLE** (Prompt 41) |
| `results/final_trading_validation/` | **IMMUTABLE** (Prompt 42) |
| `results/final_holdout_validation/` | Final holdout outputs (current frontier) |

---

## Data constraints

- Symbol: **BTCUSDT** (single-market; no external second-market transfer test in P42)
- Datasets: prefer `btcusdt_intraday_{1m,5m,15m,30m,1h}@1.0.1`
- History: **2019-01-01 → 2026-07-31** (~3.98M 1m bars); quality MINOR_GAPS
- Grade: **RESEARCH_GRADE** — not institutional (no L2/bid-ask)
- Holdout: Vision July ZIP bars after `@1.0.1` truncation (`data/btcusdt/holdout/`)

---

## Code map

| Concern | Location |
|---------|----------|
| Final holdout | `iqrp/app/backtesting/final_holdout/` |
| P42 final validation | `iqrp/app/backtesting/final_validation/` |
| P41 portfolio integration | `iqrp/app/backtesting/portfolio_integration/` |
| P40 consolidation | `iqrp/app/backtesting/alpha_research/consolidation/` |
| P39 campaign | `iqrp/app/backtesting/alpha_research/model_campaign/` |
| Unified cascade | `iqrp/app/backtesting/unified_pipeline/` |
| Binance Vision timestamps | `iqrp/app/data/historical/binance_vision.py` |
| Tests | `iqrp/tests/unit/backtesting/test_final_holdout_validation.py`, `test_final_trading_validation.py` |

Env: prefer `.venv/bin/python`.

---

## What a next agent should / should not do

**Should (only if a new prompt authorizes):**

- Longer post-P42 holdout once August+ Vision/REST data is available
- Keep claim distinctions; do not promote to LIVE_READY or PAPER_TRADING_CANDIDATE without adequate sample

**Should not (until explicitly authorized):**

- Broker / live orders
- Retuning frozen candidates after holdout
- Relaxing gates because the holdout was short
- Overwriting Prompt 35–42 artifacts
- Claiming absolute proven profitability

---

## One-line pointer

> **Full handover:** `results/ARCHITECTURAL_ROADMAP_AND_HANDOVER.md`.  
> **Resume frontier:** `results/paper_trading_validation/final_report.md`.  
> **Freeze:** research ≤2024-12-31; candidates A/B/C immutable; no paper-driven retune.  
> **Paper status:** `PAPER_TRADING_CANDIDATE` (assumed microstructure).  
> **LIVE_READY:** NO. **No broker.**

