# Result Storage

Persistence layout for operational Phase 13 backtest artifacts.

---

## Purpose

After a successful cascade, `persist_result` writes a complete audit tree under `{output_dir}/{backtest_id}/` so orders, fills, trades, positions, portfolio states, risk, performance, diagnostics, and config can be reviewed without re-running.

**Module:** `iqrp.app.backtesting.runner.persistence`  
**Related:** [BacktestReports](BacktestReports.md) · [BacktestRunner](BacktestRunner.md) · [Reproducibility](Reproducibility.md)

---

## Root layout

```text
{output_dir}/
└── {backtest_id}/
    ├── capital.json
    ├── checkpoint.json          # when pause/resume used
    ├── configuration/
    │   └── config.json
    ├── data/
    │   └── summary.json         # load_market_frame detail / validation
    ├── orders/
    │   └── orders.parquet|csv|json
    ├── fills/
    │   └── fills.*
    ├── trades/
    │   └── trades.*
    ├── positions/
    │   └── positions.*
    ├── portfolio/
    │   ├── snapshots.*
    │   └── equity_curve.json
    ├── risk/
    │   └── risk.json
    ├── performance/
    │   └── performance.json
    ├── execution/
    │   └── execution.json
    ├── walk_forward/
    │   └── walk_forward.json
    ├── scenarios/
    │   └── scenarios.json
    ├── diagnostics/
    │   ├── diagnostics.json
    │   └── reconciliation.json
    └── reports/
        ├── result.json          # full OperationalBacktestResult
        ├── report.md            # human report (write_reports)
        └── report.json          # machine report payload
```

`ARTIFACT_DIRS` creates empty directories even when optional sections are unused.

---

## Table writing

`_write_table` prefers Parquet; on failure falls back to CSV; empty tables write `[]` JSON.

---

## Checkpoint files

`checkpoint_path(root, backtest_id)` → `{root}/{backtest_id}/checkpoint.json` containing `backtest_id`, `seed`, and serialized `PipelineContext` (`to_checkpoint` / `load_checkpoint`).

---

## Locating results

```bash
# after synthetic_demo.yaml
ls results/synthetic_demo/reports/
# report.md  report.json  result.json
```

Programmatic:

```python
from pathlib import Path
from iqrp.app.backtesting.runner import BacktestRunner

runner = BacktestRunner("iqrp/configs/backtesting/synthetic_demo.yaml")
runner.validate(); runner.prepare(); runner.run()
root = Path(runner.config.output_dir) / runner.config.backtest_id
print(root)
print(runner.report())
```

---

## Critical rules

- Persist before treating a run as institutional evidence (`results_persisted` integrity check).
- Store config + data summary + seed with every run.
- Do not overwrite unrelated `backtest_id` trees; choose unique ids for experiments.
- Tables may be Parquet or CSV depending on environment codecs — both are valid artifacts.
