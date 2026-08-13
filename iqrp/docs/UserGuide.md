# User Guide — Operational Phase 13 Backtesting

Exact steps to prepare data, register, validate, configure, run, and read results for the operational institutional backtest runner.

> **Phase numbering:** Execution is **Phase 12**. This guide covers the **Phase 13** operational layer (`BacktestRunner` + data pipeline).  
> The platform **does not download market data**. You supply validated local Parquet/CSV.  
> Reference config `example_nifty50.yaml` is **wiring only** — not evidence of profitability and not an investment recommendation.

**Related:** [BacktestRunner](BacktestRunner.md) · [DataPipeline](DataPipeline.md) · [BacktestConfiguration](BacktestConfiguration.md) · [ResultStorage](ResultStorage.md) · [BacktestReports](BacktestReports.md)

---

## Prerequisites

- Repository checkout with `.venv` installed (project dependencies including pytest, pyarrow, pandas).
- Working directory: repository root (`/home/ashish/qtb` or equivalent).

---

## Commands (quick reference)

```bash
# unit + integration
.venv/bin/pytest iqrp/tests/unit/backtesting iqrp/tests/unit/backtesting/operational iqrp/tests/integration/backtesting -q --no-cov

# synthetic backtest
.venv/bin/python -m iqrp.app.backtesting.run --config iqrp/configs/backtesting/synthetic_demo.yaml

# register + run real dataset (user supplies path)
.venv/bin/python - <<'PY'
from pathlib import Path

from iqrp.app.backtesting.data import DatasetRegistry, ParquetAdapter
from iqrp.app.backtesting.runner import BacktestRunner, BacktestRunConfig
from iqrp.app.backtesting.strategy import (
    BuyAndHoldStrategy,
    CrossSectionalMomentumStrategy,
    StrategyRegistry,
)

for cls in (BuyAndHoldStrategy, CrossSectionalMomentumStrategy):
    StrategyRegistry.register(cls, overwrite=True)

# USER SUPPLIES THIS PATH — file is not downloaded by the platform
path = Path("/path/to/your/ohlcv.parquet")
assert path.exists(), f"missing user dataset: {path}"

reg = DatasetRegistry("dataset_registry.json")
rec = reg.register_file(
    path,
    dataset_id="user_supplied",
    version="1.0.0",
    canonical_parquet=True,
)
print("registered", rec.key, "checksum", rec.checksum)

adapter = ParquetAdapter(path, dataset_id=rec.dataset_id, version=rec.version)
report = adapter.validate(raise_on_critical=True)
assert report.ok, report.critical_failures
print("validation ok", report.row_count, "rows", report.instrument_count, "instruments")

cfg = BacktestRunConfig(
    backtest_id="user_historical_run",
    strategy_id="buy_and_hold",
    strategy_version="1.0.0",
    strategy_params={"mode": "equal_weight"},
    dataset_path=str(path),
    dataset_id=rec.dataset_id,
    dataset_version=rec.version,
    adapter="parquet",
    initial_capital=1_000_000.0,
    seed=42,
    output_dir="results",
    commission_bps=1.0,
    spread_bps=2.0,
    slippage_bps=1.0,
    risk_config={"max_gross_leverage": 1.0},
    meta={
        "purpose": "user_supplied_historical",
        "disclaimer": "Simulated research path only; not a profitability claim.",
    },
)
runner = BacktestRunner(cfg)
runner.validate()
runner.prepare()
result = runner.run()
print("status", runner.status().value)
print("report", runner.report())
print("equity_end", result.equity_curve[-1] if result.equity_curve else None)
PY

# operational validation report
.venv/bin/python -m iqrp.app.backtesting.operational_validation
```

---

## Step-by-step workflow

### 1. Prepare historical data

Required columns (aliases accepted — see [DataAdapters](DataAdapters.md)):

`timestamp`, `instrument`, `open`, `high`, `low`, `close`, `volume`

Timestamps must be timezone-aware UTC after load. Prefer Parquet.

**Synthetic fixture (demo / CI):**

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv

write_synthetic_ohlcv(
    Path("fixtures/synthetic_bars.parquet"),
    n_days=60,
    instruments=["AAA", "BBB"],
    seed=7,
)
print("wrote fixtures/synthetic_bars.parquet")
PY
```

Align `iqrp/configs/backtesting/synthetic_demo.yaml` `dataset_path` with that file (default: `fixtures/synthetic_bars.parquet`).

**Real user-supplied Parquet:** place your file anywhere local, e.g. `/data/nifty50_ohlcv.parquet`. Do **not** expect the platform to fetch it. If using `example_nifty50.yaml`, replace the placeholder `dataset_path` first.

### 2. Register the dataset

```python
from iqrp.app.backtesting.data import DatasetRegistry

reg = DatasetRegistry("dataset_registry.json")
rec = reg.register_file(
    "/path/to/your/ohlcv.parquet",
    dataset_id="user_supplied",
    version="1.0.0",
    canonical_parquet=True,
)
assert reg.verify_checksum(rec.dataset_id, rec.version)
```

Record `dataset_id`, `version`, and `checksum` in your experiment notes.

### 3. Validate the dataset

```python
from iqrp.app.backtesting.data import ParquetAdapter

adapter = ParquetAdapter("/path/to/your/ohlcv.parquet", dataset_id="user_supplied")
report = adapter.validate(raise_on_critical=True)
assert report.ok
```

Critical failures (naive timestamps, duplicates, invalid OHLC, …) must be fixed in the source file — not patched mid-run. See [DatasetValidation](DatasetValidation.md).

### 4. Select a strategy

Register and choose explicitly:

| ID | Use |
|----|-----|
| `buy_and_hold` | Reference equal-weight / first-instrument hold (pipeline smoke) |
| `cross_sectional_momentum` | Reference lookback rank demo |

```python
from iqrp.app.backtesting.strategy import BuyAndHoldStrategy, StrategyRegistry

StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
# strategy_id="buy_and_hold", strategy_version="1.0.0"
```

Custom strategies: subclass `Strategy`, set `strategy_id` / `strategy_version`, register before `validate()`.

### 5. Configure the backtest

Edit YAML or build `BacktestRunConfig` in Python. Minimum: `strategy_id`, `dataset_path`, `seed`, `output_dir`, `backtest_id`. See [BacktestConfiguration](BacktestConfiguration.md).

Synthetic:

```bash
# ensure fixture exists (step 1), then:
.venv/bin/python -m iqrp.app.backtesting.run --config iqrp/configs/backtesting/synthetic_demo.yaml
```

CLI overrides example:

```bash
.venv/bin/python -m iqrp.app.backtesting.run \
  --config iqrp/configs/backtesting/example_nifty50.yaml \
  --dataset /path/to/your/ohlcv.parquet \
  --strategy buy_and_hold \
  --seed 42 \
  --output results
```

(`example_nifty50.yaml` remains a reference template only.)

### 6. Run the backtest

Programmatic lifecycle:

```python
runner = BacktestRunner(cfg)
runner.validate()
runner.prepare()
result = runner.run()
assert runner.status().value in {"COMPLETED", "INVALIDATED", "FAILED"}
```

Expect `COMPLETED` only when preflight and integrity succeed. Look-ahead → `INVALIDATED`.

### 7. Locate results

```text
results/{backtest_id}/
  configuration/config.json
  orders/ fills/ trades/ positions/ portfolio/
  risk/ performance/ execution/ diagnostics/
  reports/report.md
  reports/report.json
  reports/result.json
```

Example for synthetic demo: `results/synthetic_demo/reports/report.md`.

### 8. Read the report

```bash
less results/synthetic_demo/reports/report.md
```

Read the executive summary **and** the limitations / reproducibility sections. Simulated equity and returns describe this run under stated assumptions only — not a profitability claim, including for any NIFTY50-named reference config.

---

## Tests

```bash
.venv/bin/pytest iqrp/tests/unit/backtesting iqrp/tests/unit/backtesting/operational iqrp/tests/integration/backtesting -q --no-cov
```

---

## Operational validation

```bash
.venv/bin/python -m iqrp.app.backtesting.operational_validation
```

Produces the operational validation report for the Phase 13 executable layer (component status, tests, known limitations).

---

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `dataset path not found` | Supply a real local path; nothing is downloaded |
| `preflight validation failed` / strategy not registered | Register strategy or pass `strategy=` override |
| `critical data-quality failures` | Fix source data; re-register new version |
| `INVALIDATED` | Inspect `invalidation_reason` / diagnostics for look-ahead or risk breach |
| Empty equity | Confirm date filters leave bars; check status/exceptions |

---

## Critical rules

1. User supplies data — no silent market download.  
2. Explicit `strategy_id` — no arbitrary selection.  
3. Critical data / PIT failures stop or invalidate the run.  
4. Same config + dataset checksum + seed → reproducible path.  
5. Reports are research/simulation output, not performance guarantees.  
6. `example_nifty50.yaml` ≠ profitability evidence.
