# IQRP — Complete Architectural Roadmap & Agent Handover

**Document ID:** `ARCHITECTURAL_ROADMAP_AND_HANDOVER`  
**Workspace:** `/home/ashish/qtb`  
**Remote:** `https://github.com/laksh890/QTB_curr.git`  
**Audience:** Next human or model agent continuing IQRP research (especially after `git clone`)  
**Companion (short pointer):** `results/AGENT_HANDOFF_MAP.md`  
**GitHub clone / recovery:** `results/GITHUB_CLONE_AND_RECOVERY.md` ← **read first on a fresh clone**  
**Python env:** Prefer `.venv/bin/python` or `poetry install --with dev,ml`  
**Generated for frontier:** Prompt 43 complete (`PAPER_TRADING_CANDIDATE`)

---

## 0. One-screen status (read this first)

| Item | Value |
|------|--------|
| **Clone readiness** | See blockers below — **recover data packages before new research** |
| **Current research frontier** | Prompt 43 — sequential paper trading under assumed microstructure |
| **Status** | `PAPER_TRADING_CANDIDATE` |
| **Primary report** | `results/paper_trading_validation/final_report.md` |
| **Frozen sleeves** | A=`mdc_99aa952c5d5f6ff7`, B=`mdc_6f008c954ea26bf5`, C=`mdc_678609c534d68189` |
| **Research freeze** | Definitions + selection ≤ **2024-12-31**; calendar **2025** = independent eval only |
| **LIVE_READY** | **NO** |
| **Broker / live orders** | **HARD STOP — not authorized** |
| **Do not** | Retune A/B/C from paper or 2025 results; overwrite Prompt 35–42 artifacts; claim absolute proven profit |

**Net returns in Prompt 43 are full-year 2025 cumulative** (`final_equity / 100_000 - 1`), not daily.

### Institutional clone blockers (board status)

| Priority | Blocker | Action |
|----------|---------|--------|
| **1 (first)** | `iqrp/app/data` + `iqrp/app/backtesting/data` were gitignored by bare `data/` | Fixed ignore → `/data/`; packages must be **committed + pushed** |
| **2** | No `poetry.lock` | Run `poetry lock` and commit the lockfile |
| **3** | ML libs used in code not pinned in core Poetry | Optional group `ml` in `pyproject.toml`; install `--with ml` |

**Nothing else to implement until the user picks the next slice.** Recovering both data packages for GitHub remains the first step.

Verify:

```bash
test -f iqrp/app/data/__init__.py && test -f iqrp/app/backtesting/data/dataset_registry.py
```

Details: `results/GITHUB_CLONE_AND_RECOVERY.md`.

---

## 1. Purpose of this document

This is the **authoritative long-form handover** for IQRP’s research-to-paper path. It covers:

1. End-to-end architecture (platforms + cascade)  
2. Prompt-by-prompt progress (≈35→43) and what each unlocked  
3. Feature inventory of major `iqrp/app` packages  
4. Frozen candidates, claim ladder, and evidence grades  
5. Data assets, immutability rules, and CLI entry points  
6. Gaps, risk of misinterpretation, and **authorized next phases only**

Shorter day-to-day pointer remains `results/AGENT_HANDOFF_MAP.md`. Prefer this file when a new agent needs full context.

---

## 2. What IQRP is (and is not)

**IQRP** is a research-grade quantitative platform: data → features/signals → forecasts/regimes → alpha research → portfolio/risk → simulated execution → accounting, with strict claim distinctions.

It is **not**:

- A live trading system (`iqrp/app/live` is a stub; no broker adapter wired for production)  
- Institutional market data (OHLCV only; no observed L2/bid-ask in the validated path)  
- Proof of absolute profitability (even when statuses say `PROVEN_RESEARCH_PROFITABILITY` or `PAPER_TRADING_CANDIDATE`)

---

## 3. Master architecture (as of Prompt 43)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA PLATFORM                                                           │
│  registry · provenance · quality · Yahoo · Binance Vision · resampling   │
│  BTCUSDT OHLCV @1.0.1 (2019→2026-07) · firewall 2024/2025 slices         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FEATURES / REFERENCE SIGNALS                                            │
│  FeatureRegistry · SignalRegistry · momentum/RSI/vol/volume/…            │
│  MTF causal construction (no future bars)                                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FORECASTING / REGIMES (research library; adapters into signals)         │
│  GARCH · ARIMA/VAR/… · XGB/LGBM/CatBoost · LSTM/GRU/… · TiDE/… · HMM/… │
│  Prompt 37: forecast_adapter / regime_adapter → SignalRegistry (opt-in)  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ALPHA RESEARCH (Prompts 35–40)                                          │
│  leakage · IC · costs · OOS · ranking · model campaign · consolidation   │
│  Output: AlphaCandidate + definition checksums (immutable thereafter)    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO + RISK (Prompts 38, 41)                                       │
│  constraints · TargetWeights · optimizers (MV/RP/BL/HRP) · risk gates    │
│  RiskIntelligenceEngine · position limits · kill switches (paper)        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  EXECUTION + ACCOUNTING (Prompt 38 cascade; Prompt 43 sequential paper)  │
│  UnifiedTradingOrchestrator · ExecutionEngine · SimulatedVenue           │
│  AssumedFillModel (OHLCV microstructure) · ledgers · reconciliation      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  VALIDATION LADDER                                                       │
│  P42 deep validate → P41 portfolio → firewall 2024/2025 → P43 paper sim  │
│  Artifacts under results/*  · statuses ≠ LIVE_READY                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Sequential paper path (Prompt 43)

```
bar[t]
  → update available info (≤ t only)
  → frozen signal / target weight
  → risk limits + kill switch
  → portfolio target (sleeve / combo)
  → order (+ latency bars)
  → AssumedFillModel (spread/slip/commission/partial/reject)
  → position book + cash
  → accounting mark
  → reconcile (drift → halt further orders)
  → paper performance
```

**Cost label:** `ASSUMED_OHLCV_MICROSTRUCTURE` — not fabricated historical bid/ask; spreads/slippage are **assumed** from mid.

---

## 4. Claim ladder (never collapse)

```
MODEL IMPLEMENTED
≠ FORECAST
≠ SIGNAL
≠ BACKTESTABLE
≠ SURVIVES OOS
≠ SURVIVES COSTS
≠ ROBUST
≠ DISTINCT AFTER CONSOLIDATION          ← Prompt 40
≠ PROFITABILITY_EVIDENCE                ← Prompt 42 (3 MTF IDs)
≠ HOLDOUT REPLICATED (calendar 2025)    ← frozen_2024_2025_holdout
≠ PROVEN_RESEARCH_PROFITABILITY         ← research evidence only
≠ PAPER_SIMULATION_OPERATIONAL
≠ PAPER_VALIDATION_PASS / WEAK
≠ PAPER_TRADING_CANDIDATE               ← Prompt 43 (current max for paper)
≠ PROVEN PROFITABLE (absolute)          ← NEVER from sim alone
≠ LIVE_READY / PRODUCTION_READY         ← NEVER from current work
```

Statuses **allowed** at paper stage:  
`PAPER_SIMULATION_OPERATIONAL` | `PAPER_VALIDATION_WEAK` | `PAPER_VALIDATION_PASS` | `PAPER_TRADING_CANDIDATE`

Statuses **forbidden** as conclusions from Prompt 43 alone:  
`PROVEN PROFITABLE` | `LIVE_READY` | `PRODUCTION_READY`

---

## 5. Prompt roadmap — completed progress

| Prompt / phase | Intent | Code | Results | Outcome |
|----------------|--------|------|---------|---------|
| **~01–12 platforms** | Data, features, forecast, risk, portfolio, execution, backtest libs | `iqrp/app/{data,features,forecasting,risk,portfolio,execution,backtesting,…}` | Architecture audits | Libraries **COMPLETE**; none production-ready |
| **35** | Full BTC alpha research (reference signals) | `alpha_research/` | `results/alpha_research_btc_full/` | **IMMUTABLE** campaign baseline |
| **36** | Statistical / leakage audit | audit packages | `results/alpha_research_btc_full_audit/` | Validity **LIMITED**; constraints carry forward |
| **37** | Model → signal adapters | `alpha_research/adapters/` | `results/model_alpha_integration/` | Model-driven signals operational |
| **38** | Unified trading cascade | `backtesting/unified_pipeline/` | `results/unified_trading_pipeline/` | Cascade **OPERATIONAL** (sim) |
| **Arch audit** | End-to-end architecture evidence | `unified_pipeline/final_architecture_audit.py` | `results/final_system_architecture_audit/` | **ARCHITECTURE COMPLETE**; not live |
| **39** | Model-driven alpha campaign | `model_campaign/` | `results/model_driven_alpha_campaign/` | Large experiment registry; **IMMUTABLE** defs |
| **40** | Candidate consolidation | `consolidation/` | `results/candidate_consolidation/` | 8 distinct + ensembles; **IMMUTABLE** set for P42 |
| **41** | Portfolio construction integration | `portfolio_integration/` | `results/portfolio_construction_integration/` | Optimizers on frozen alphas; **STOP** before paper at time |
| **42** | Final trading validation (extended BTC) | `final_validation/` | `results/final_trading_validation/` | **3×** `PROFITABILITY_EVIDENCE` (MTF family) |
| **Post-P42 short holdout** | Bars after `@1.0.1` trunc | `final_holdout/`, `independent_validation/` | `final_holdout_validation/`, `independent_candidate_validation/` | **INVALID / WEAK** (~1 day); not replication |
| **Frozen 2024→2025** | True calendar firewall | `frozen_2025_holdout/` | `results/frozen_2024_2025_holdout/` | 2× `PROVEN_RESEARCH_PROFITABILITY`, 1× `PAPER_TRADING_CANDIDATE` (research) |
| **43** | Sequential paper + realistic assumed fills | `iqrp/app/paper_trading/` | `results/paper_trading_validation/` | **`PAPER_TRADING_CANDIDATE`**; **STOP** no broker |

### Prompt 42 evidence trio (later frozen as A/B/C)

| ID | Family | TF / dir (approx) | P42 class |
|----|--------|-------------------|-----------|
| `mdc_99aa952c5d5f6ff7` | MTF momentum | 15m LONG_SHORT | `PROFITABILITY_EVIDENCE` |
| `mdc_6f008c954ea26bf5` | MTF momentum | 15m LONG | `PROFITABILITY_EVIDENCE` |
| `mdc_678609c534d68189` | MTF momentum | 5m SHORT | `PROFITABILITY_EVIDENCE` |

### Frozen 2024→2025 holdout (research evidence)

| Label | ID | 2025 research status | ADVERSE |
|-------|-----|----------------------|---------|
| A | `mdc_99aa952c5d5f6ff7` | `PROVEN_RESEARCH_PROFITABILITY` | survives |
| B | `mdc_6f008c954ea26bf5` | `PROVEN_RESEARCH_PROFITABILITY` | survives |
| C | `mdc_678609c534d68189` | `PAPER_TRADING_CANDIDATE` | fails |

### Prompt 43 paper sim (BASE assumed costs, full year 2025)

| Combo | Net return (year) | Recon | Notes |
|-------|------------------:|-------|-------|
| A | ~+60.8% | ok | Strongest single sleeve under BASE |
| B | ~+27.5% | ok | |
| C | ~+50.9% | ok | |
| A+B | ~+60.8% | ok | ≈ A; little diversification gain |
| A+C / B+C | worse Sharpe | ok | Mixed-TF asof-aligned; not clearly better |
| A+B+C | ~+60.7% | ok | ≈ A |

**Cost sensitivity (A):** BASE ~+61% → MODERATE ~+35% → **ADVERSE ~−23%**. Paper P&L is **not** robust to harsh assumed costs.

**Paper Sharpes** are path-dependent under assumed fills + bar annualization; do **not** equate 1:1 with research-batch Sharpes (~5–7).

---

## 6. Platform feature inventory (capabilities present)

### 6.1 Data (`iqrp/app/data`)

- Historical providers: Yahoo Finance, Binance Vision  
- Registry + provenance + quality reports  
- Resampling, calendars, timestamp normalization (incl. µs Vision fix)  
- Intraday BTCUSDT parquets + firewall slices under `data/btcusdt/`  
- **Grade:** RESEARCH_GRADE OHLCV — not L2/tick institutional

### 6.2 Features / signals (`iqrp/app/features`, SignalRegistry)

- Momentum, statistical, volume, microstructure (research), calendar, cross-asset hooks  
- Feature store / metadata / research utilities  
- Reference OHLCV signals used heavily in Prompt 35+

### 6.3 Forecasting (`iqrp/app/forecasting`)

- Statistical: AR/MA/ARIMA/SARIMA/VAR/VECM/…  
- Volatility: GARCH-family  
- Trees: XGBoost / LightGBM / CatBoost  
- Neural: LSTM/GRU/MLP/TCN/N-HiTS/DeepAR/…  
- Transformers: TiDE / Informer-family utilities  
- Pre/post-processing, diagnostics, explainability, serialization  
- **Integrated to trading via adapters (P37), not every model is a live alpha**

### 6.4 Regimes / state (`iqrp/app/regimes`, `state_space`)

- HMM, Markov, GMM, Kalman, Bayesian hooks  
- State-space models / smoothing / forecasting  
- Default alpha path often uses reference signals; regimes available via adapters

### 6.5 Labels / timeseries analytics

- Classification/regression/volatility/regime/meta labels  
- Stationarity, decomposition, change points, spectral, wavelets, etc.  
- Available as analytics; not required on every paper path

### 6.6 Risk (`iqrp/app/risk`)

- Limits: position, exposure, loss, concentration, liquidity  
- Sizing, capital, tail, stress, VaR-style tooling, monitoring  
- Unified pipeline uses gate + sizing; full VaR/MC not mandatory per candidate  
- Paper layer: `iqrp/app/paper_trading/risk.py` kill switches

### 6.7 Portfolio (`iqrp/app/portfolio`)

- Construction, covariance, expected returns, constraints  
- Optimizers: mean-variance, risk parity, Black–Litterman, HRP, robust hooks  
- Prompt 41 validated integration with frozen alphas  
- Paper combos use **sleeve weight sum + gross clip** (candidates unmodified)

### 6.8 Execution (`iqrp/app/execution`)

- ExecutionEngine, order manager, simulated venue, latency, slippage, TCA  
- Smart routing / algorithms (research/sim)  
- **No live brokerage integration authorized**

### 6.9 Backtesting (`iqrp/app/backtesting`)

- Event engine, walk-forward, scenarios, accounting, performance, horizon  
- Alpha research + model campaign + consolidation  
- Unified pipeline orchestrator  
- Final validation / holdout / frozen 2025 / independent validation packages  
- Rolling retraining utilities (research)

### 6.10 Paper trading (`iqrp/app/paper_trading`) — Prompt 43

| Module | Role |
|--------|------|
| `protocol.py` | Frozen IDs, EXEC_SCENARIOS, status classifier, disclaimer |
| `fill_model.py` | Assumed spread/slip/commission/partial/reject |
| `simulator.py` | Bar-by-bar sequential session + recon |
| `risk.py` | Limits + kill switch |
| `failure_injection.py` | Operational failure battery |
| `runner.py` | Full validation + artifact writer |
| `__main__.py` | CLI |

### 6.11 Other

- `dashboard/`, `api/`, `simulation/`, `math/`, `core/`, `config/` — supporting platforms  
- `live/` — **not** production-ready; do not treat as broker-ready

---

## 7. Unified cascade (Prompt 38) — stage matrix

Validated as **PASS** for reference and model signals (sim):

Data → Model → Forecast/Regime → Adapter → SignalRegistry → Alpha → Candidate → Risk → Sizing → Portfolio → Orders → Execution → Fills → Positions → Accounting → Reconciliation → Performance

**Meaning:** The plumbing works. It does **not** mean strategies are profitable or live-ready.

Orchestrator entry: `iqrp/app/backtesting/unified_pipeline/orchestrator.py`  
Validate: `…/validate_pipeline.py`

---

## 8. Data assets & firewall

### Primary registered BTC

| Dataset | Version | Span (approx) | Role |
|---------|---------|---------------|------|
| `btcusdt_intraday_{1m,5m,15m,30m,1h}` | **@1.0.1** | 2019-01 → 2026-07 | Extended research history |
| Quality | RESEARCH_GRADE | MINOR_GAPS | OHLCV only |

Parquets live under `data/btcusdt/`. Prefer `@1.0.1` checksums in registry.

### Temporal firewall (authoritative for A/B/C eval)

| Slice | Path / registry | Rule |
|-------|-----------------|------|
| Research | `data/btcusdt/firewall_2024_2025/btcusdt_research_through_2024_*.parquet` | ≤ 2024-12-31 |
| Holdout | `…/btcusdt_holdout_2025_*.parquet` | Calendar 2025 only |
| Materializer | `iqrp/app/backtesting/frozen_2025_holdout/datasets.py` | Do not mix for selection |

**1m holdout completeness:** 525,600 bars (full 2025).

### Short post-P42 remnant (do not confuse with 2025 firewall)

- `data/btcusdt/holdout/` Vision bars after `@1.0.1` truncation (~1 day)  
- Drove `INVALID_HOLDOUT` / `WEAK_EVIDENCE` — **superseded** by calendar-2025 firewall for evidence claims

---

## 9. Frozen candidates — immutable contract

```
FROZEN_CANDIDATES = {
  "A": "mdc_99aa952c5d5f6ff7",
  "B": "mdc_6f008c954ea26bf5",
  "C": "mdc_678609c534d68189",
}
```

**Must not change:**

- Kind / source_id / timeframe / direction / holding_bars / parameters  
- Definition checksums (see `results/frozen_2024_2025_holdout/frozen_candidate_manifest.json` and Prompt 39 experiment JSON)  
- Selection based on 2025 or paper results  

**Combinations (P43):** Sum sleeves with gross clip; causal `merge_asof` for mixed TF onto primary TF. Candidates themselves unchanged.

**Definition source of truth:** Prompt 39 experiments under `results/model_driven_alpha_campaign/` via `load_p39_experiment`.

---

## 10. Results artifact map (authoritative directories)

| Directory | Role | Mutability |
|-----------|------|------------|
| `results/alpha_research_btc_full/` | P35 baseline | **IMMUTABLE** |
| `results/alpha_research_btc_full_audit/` | P36 audit | **IMMUTABLE** |
| `results/model_alpha_integration/` | P37 | Prefer immutable |
| `results/unified_trading_pipeline/` | P38 | Prefer immutable |
| `results/final_system_architecture_audit/` | Arch verdict | Prefer immutable |
| `results/model_driven_alpha_campaign/` | P39 defs | **IMMUTABLE** |
| `results/candidate_consolidation/` | P40 set | **IMMUTABLE** |
| `results/portfolio_construction_integration/` | P41 | **IMMUTABLE** |
| `results/final_trading_validation/` | P42 | **IMMUTABLE** |
| `results/final_holdout_validation/` | Short remnant holdout | Historical; weak sample |
| `results/independent_candidate_validation/` | Short remnant / INVALID | Historical |
| `results/frozen_2024_2025_holdout/` | True 2025 research holdout | Prefer immutable |
| `results/paper_trading_validation/` | **P43 current frontier** | Current outputs |
| `results/paper_trading_validation_smoke/` | Smoke only | Disposable |
| `results/AGENT_HANDOFF_MAP.md` | Short pointer | Keep in sync |
| **This file** | Long handover | Keep in sync |

---

## 11. Code map & CLI cheatsheet

```bash
# Env
cd /home/ashish/qtb
.venv/bin/python -m …

# Prompt 43 (current)
.venv/bin/python -m iqrp.app.paper_trading
.venv/bin/python -m iqrp.app.paper_trading --smoke

# Frozen 2024→2025 research holdout
.venv/bin/python -m iqrp.app.backtesting.frozen_2025_holdout

# Prompt 42
.venv/bin/python -m iqrp.app.backtesting.final_validation
.venv/bin/python -m iqrp.app.backtesting.final_validation --smoke

# Prompt 41 / 40 / 39 / 38 (historical re-runs — do not overwrite without authorization)
.venv/bin/python -m iqrp.app.backtesting.portfolio_integration
.venv/bin/python -m iqrp.app.backtesting.alpha_research.consolidation   # if exposed
.venv/bin/python -m iqrp.app.backtesting.alpha_research.model_campaign
.venv/bin/python -m iqrp.app.backtesting.unified_pipeline.validate_pipeline

# Unit tests (examples)
.venv/bin/python -m pytest iqrp/tests/unit/backtesting/test_paper_trading_validation.py -q --no-cov
.venv/bin/python -m pytest iqrp/tests/unit/backtesting/test_final_trading_validation.py -q --no-cov
```

| Concern | Path |
|---------|------|
| GitHub clone recovery | `results/GITHUB_CLONE_AND_RECOVERY.md` |
| App data platform | `iqrp/app/data/` |
| DatasetRegistry / BT adapters | `iqrp/app/backtesting/data/` |
| Paper trading | `iqrp/app/paper_trading/` |
| Frozen 2025 holdout | `iqrp/app/backtesting/frozen_2025_holdout/` |
| Final validation P42 | `iqrp/app/backtesting/final_validation/` |
| Portfolio integration P41 | `iqrp/app/backtesting/portfolio_integration/` |
| Consolidation P40 | `iqrp/app/backtesting/alpha_research/consolidation/` |
| Model campaign P39 | `iqrp/app/backtesting/alpha_research/model_campaign/` |
| Unified pipeline P38 | `iqrp/app/backtesting/unified_pipeline/` |
| Cost scenarios | `iqrp.app.backtesting.alpha_research.types.COST_SCENARIOS` |
| Binance Vision | `iqrp/app/data/historical/binance_vision.py` |

---

## 12. Lineage (required mental model)

```
dataset (registry / firewall)
  → model / reference construction (≤ research cut)
  → signal (causal)
  → AlphaCandidate (frozen id + checksum)
  → portfolio target (sleeve / combo)
  → risk decision / kill switch
  → order (+ latency)
  → fill (assumed microstructure)
  → position
  → accounting / equity
  → performance + recon
```

Every Prompt 43 fill records signal/order/fill timestamps, qty, prices, fees, slippage, spread assumption, latency, status.

---

## 13. Hard stops & policy

Until a **new explicit prompt** authorizes otherwise:

1. **No broker connection**  
2. **No live orders**  
3. **No optimizing / retraining / selecting** A/B/C from 2025 or paper results  
4. **No relaxing gates** because a holdout was short  
5. **No overwriting** Prompt 35–42 result trees  
6. **No claim** of absolute proven profitability or LIVE_READY from simulation  
7. Paid L2/tick → **STOP_BEFORE_PURCHASE** (survey only unless authorized)

---

## 14. Known gaps / honesty checklist (for next agent)

| Gap | Implication |
|-----|-------------|
| No observed bid/ask | Fills are assumed; ADVERSE can erase paper P&L |
| Single market BTCUSDT | No cross-market transfer proof |
| Paper sim scale-free | $100 vs $100k same % in sim; real min notional not modeled |
| Mixed-TF combos | Causal asof onto primary TF — approximate diversification |
| High paper Sharpes | Do not treat as institutional Sharpe truth |
| `live/` stub | Not sandbox-broker integrated |
| ML controls on 2025 | Largely rejected in frozen holdout — do not revive without new protocol |
| Short post-P42 remnant | Invalid as replication; ignore for promotion |

---

## 15. Authorized roadmap — possible next phases

**Gate:** Do not start these until GitHub clone blockers 1–2 are cleared (both `data` packages on remote + preferably `poetry.lock`).

Only proceed if a **new user prompt** specifies which phase:

| Phase | Goal | Prerequisite |
|-------|------|--------------|
| **P44a — Broker sandbox design (no orders)** | Adapter interfaces, auth, symbol maps, dry-run schemas | Keep LIVE_READY=NO |
| **P44b — Observed microstructure paper** | Replace assumed spread with purchased/free L2 if authorized | STOP_BEFORE_PURCHASE until approved |
| **P44c — Capacity / min-notional study** | Retail capital floors ($100–500 realism) | Additive; no retune |
| **P44d — Multi-asset / second market** | Transfer test | New data + firewall |
| **P44e — Longer post-2025 holdout** | 2026+ OOS once complete local data | No retune on peek |
| **Ops hardening** | Restart recovery, monitoring, alert hooks on paper sim | Still no broker |

**Default if unclear:** STOP. Ask the user which phase. Do not invent live trading.

---

## 16. Suggested onboarding sequence for a new agent

0. If this is a **GitHub clone**: read `results/GITHUB_CLONE_AND_RECOVERY.md` and verify both data packages + Poetry lock status.  
1. Read **this file** (architecture + progress).  
2. Skim `results/AGENT_HANDOFF_MAP.md`.  
3. Read `results/paper_trading_validation/final_report.md` + `paper_trading_status.json`.  
4. Read `results/frozen_2024_2025_holdout/final_report.md` + `firewall_audit.json`.  
5. Skim `results/final_trading_validation/final_report.md` (P42 gates).  
6. Skim `results/final_system_architecture_audit/final_report.md` (platform matrix).  
7. Open frozen IDs in Prompt 39 / manifest — confirm checksums before any code change.  
8. Wait for user authorization before any new phase.

---

## 17. Acceptance gates already satisfied (Prompt 43)

- Frozen candidates unchanged; no 2025 retune  
- Sequential processing; no lookahead by construction  
- Assumed realistic execution model operational  
- Zero unexplained recon drift (fills ↔ positions ↔ equity)  
- Risk limits + kill switches + failure injection PASS  
- Restart scenario PASS; reproducibility PASS  
- Status: `PAPER_TRADING_CANDIDATE` with `LIVE_READY: false`

---

## 18. Final answers (Prompt 43 research question)

> Can the alpha and portfolio architecture that survived 2024→2025 research behave like a trading system under sequential realistic simulated conditions?

**Yes — as a paper simulation** under assumed microstructure, with working risk/kill/recon, and BASE-cost survivors A/B/C.  

**Not** proven for live markets, **not** robust to ADVERSE assumed costs, **not** broker-integrated.

---

## 19. Document maintenance

When a new prompt completes:

1. Update **section 0** and **section 5** table.  
2. Point `results/AGENT_HANDOFF_MAP.md` “Start here” at the new frontier.  
3. Add results directory to **section 10**.  
4. Refresh claim ladder if new statuses are introduced (never collapse).  

---

**STOP.** No broker. No live orders. No candidate retuning. Resume only with an explicit next prompt.
