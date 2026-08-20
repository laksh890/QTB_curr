# GitHub Clone Blockers & Recovery Guide

**Audience:** Any model/agent cloning `https://github.com/laksh890/QTB_curr` (or forks).  
**Companion:** `results/ARCHITECTURAL_ROADMAP_AND_HANDOVER.md`  
**Policy:** No broker / live orders. Do not retune frozen A/B/C.

---

## Institutional status — open blockers

| # | Blocker | Severity | Status on fresh GitHub clone (before this recovery commit) | First action |
|---|---------|----------|------------------------------------------------------------|--------------|
| **1** | `iqrp/app/data` missing from git | **CRITICAL** | Entire package ignored by bare `data/` rule in `.gitignore` | Recover/commit package (this slice) |
| **2** | `iqrp/app/backtesting/data` missing from git | **CRITICAL** | Same `data/` ignore → no `DatasetRegistry`, adapters, validators | Recover/commit package (this slice) |
| **3** | No `poetry.lock` | **HIGH** | Non-reproducible installs; transitive deps float | `poetry lock` + commit lockfile |
| **4** | ML libs used in code not pinned in Poetry core | **HIGH** | `xgboost` / `lightgbm` / `catboost` / `torch` / `sklearn` imported but absent from locked deps | Install `--with ml` after lock; keep group pinned |

**Nothing else to implement until these are cleared.** Research frontier (Prompt 43) stays paused for clone readiness.

---

## Root cause (blocker 1–2)

`.gitignore` previously contained:

```gitignore
data/
```

Git treats that as **any path segment named `data/`**, which excluded:

- `/data/` — intended (market parquets; stay local)
- `iqrp/app/data/` — **unintended** (historical providers, Binance Vision, acquire/validate)
- `iqrp/app/backtesting/data/` — **unintended** (`DatasetRegistry`, parquet/csv adapters, schema, PIT, universe)

Local workspaces that already had those trees looked “complete.” Fresh clones from GitHub did **not**.

### Fix applied

```gitignore
/data/                          # root market data only
!iqrp/app/data/
!iqrp/app/data/**
!iqrp/app/backtesting/data/
!iqrp/app/backtesting/data/**
```

After the recovery commit lands on `origin/main`, a clone must contain both packages.

---

## Verify after `git clone` / `git pull`

```bash
cd QTB_curr   # or your clone path
test -f iqrp/app/data/__init__.py && echo OK_app_data || echo MISSING_app_data
test -f iqrp/app/backtesting/data/dataset_registry.py && echo OK_bt_data || echo MISSING_bt_data
test -f poetry.lock && echo OK_lock || echo MISSING_lock
```

Expected imports:

```bash
.venv/bin/python -c "from iqrp.app.backtesting.data import DatasetRegistry; print('DatasetRegistry OK')"
.venv/bin/python -c "import iqrp.app.data; print('iqrp.app.data OK')"
```

---

## Recovery procedure (if still missing after pull)

### A. Packages present on a donor machine but not on GitHub

1. Confirm ignore fix: `git check-ignore -v iqrp/app/data/__init__.py` → **must print nothing**.  
2. Stage sources only (no `__pycache__`, no `/data` parquets):

```bash
git add .gitignore pyproject.toml
git add iqrp/app/data iqrp/app/backtesting/data
git status  # confirm .py files staged; no data/btcusdt/*.parquet
```

3. Commit + push to the shared remote (see section “Push checklist”).

### B. Fresh clone still empty (commit not pushed yet)

Copy from an authorized local workspace that has the packages, then commit from that machine — do **not** invent DatasetRegistry from scratch if the donor tree exists.

### C. Market data (`/data/btcusdt/…`) — separate from Python packages

Root `/data/` remains gitignored. Clones need local acquisition:

- Prefer registered BTCUSDT `@1.0.1` + firewall slices under `data/btcusdt/firewall_2024_2025/`  
- Use existing historical acquisition / Vision pipeline once `iqrp.app.data.historical` is importable  
- Do **not** commit multi‑GB parquets unless explicitly authorized

---

## Poetry lock + ML pins (blockers 3–4)

### Current intent in `pyproject.toml`

- Core deps: numeric/data stack (+ `pandas` declared for research paths)  
- Optional group **`ml`**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch`  
- Install: `poetry install --with dev,ml`

### Required next commands (after packages are on GitHub)

```bash
# If poetry is not installed: pipx install poetry   OR   apt/brew install poetry
poetry lock                 # creates poetry.lock — MUST be committed
poetry install --with dev,ml
```

**Note (this workspace):** `poetry` was not on PATH when packages were recovered; `poetry.lock` may still be absent after the recovery commit. Treat lockfile generation as the **immediate follow-up** for the next agent with Poetry available.

- Do not claim reproducible environments  
- Prefer documenting exact versions used on the research machine (see roadmap)  
- Do not silently `pip install` untracked versions for “PASS” CI

Known research-venv reference versions (informational; lock may resolve slightly differently):

| Package | Reference venv |
|---------|----------------|
| xgboost | 3.4.0 |
| lightgbm | 4.7.0 |
| catboost | 1.2.10 |
| torch | 2.13.0+cpu |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |

---

## What a GitHub-pulled agent must / must not do

**Must (first slice):**

1. Confirm both data packages exist post-pull  
2. If missing → recover/commit (do not start Prompt 44+)  
3. Generate + commit `poetry.lock` if still absent  
4. `poetry install --with dev,ml` before ML campaign reruns  

**Must not:**

- Treat Prompt 43 paper status as LIVE_READY  
- Retune frozen candidates A/B/C  
- Connect a broker  
- Commit secrets / `.env` / root `/data` blobs  
- Re-implement DatasetRegistry if the recovered package is available  

**Resume research frontier only after:** packages present + lockfile committed (or explicitly deferred by user).

---

## Push checklist (human or agent with push rights)

Local recovery commit (example): `bcf2fb6` on `main` — **ahead of origin until pushed**.

```bash
git status   # expect: ahead of origin/main by N
git check-ignore -v iqrp/app/data/__init__.py iqrp/app/backtesting/data/dataset_registry.py
# both should NOT be ignored for staging purposes (negation rules OK)

git push -u origin HEAD
```

If HTTPS auth fails (`could not read Username`), use SSH remote or `gh auth login`, then push.

---

## Linkage to research frontier

| Item | Location |
|------|----------|
| Full architecture + Prompt 35–43 | `results/ARCHITECTURAL_ROADMAP_AND_HANDOVER.md` |
| Short pointer | `results/AGENT_HANDOFF_MAP.md` |
| Paper frontier | `results/paper_trading_validation/final_report.md` |
| Frozen A/B/C | `mdc_99aa952c5d5f6ff7`, `mdc_6f008c954ea26bf5`, `mdc_678609c534d68189` |

**STOP after recovery.** Next research slice only when the user picks it.
