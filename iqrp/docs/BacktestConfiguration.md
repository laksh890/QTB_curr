# Backtest Configuration

Operational run configuration for Phase 13 (`BacktestRunConfig`): YAML, dict, or OmegaConf.

---

## Purpose

Describe a single executable backtest: strategy identity, local dataset reference, capital/costs, risk/portfolio/execution knobs, seeds, output paths, and optional research extensions.

**Module:** `iqrp.app.backtesting.runner.configuration`  
**Type:** `BacktestRunConfig`  
**Configs:** `iqrp/configs/backtesting/`  
**Related:** [BacktestRunner](BacktestRunner.md) · [UserGuide](UserGuide.md) · [Reproducibility](Reproducibility.md)

> Distinct from research Hydra `BacktestSettings` (`iqrp.app.backtesting.config` / `default.yaml`), which configures the broader `BacktestEngine` platform. Operational runs use `BacktestRunConfig`.

---

## Fields

| Field | Default | Role |
|-------|---------|------|
| `backtest_id` | `"backtest"` | Results subdirectory name |
| `strategy_id` | `""` | Required; registry key |
| `strategy_version` | `"1.0.0"` | Registry version |
| `strategy_params` | `{}` | Kwargs to strategy constructor |
| `dataset_id` / `dataset_version` | `None` | Lineage labels |
| `dataset_path` | `None` | **Required for load** (local path) |
| `adapter` | `"parquet"` | `parquet` or `csv` |
| `universe` | `[]` | Optional instrument filter |
| `start` / `end` | `None` | Inclusive date window (end expands to EOD if date-only) |
| `frequency` | `"daily"` | Clock frequency alias |
| `initial_capital` | `1_000_000` | Alias: `capital` |
| `currency` / `timezone` | `USD` / `UTC` | Accounting / clock |
| `seed` | `42` | Stochastic execution paths |
| `output_dir` | `"results"` | Root for artifacts |
| `commission_bps` / `spread_bps` / `slippage_bps` / `financing_bps` | cost knobs | |
| `risk_config` | `{}` | e.g. `max_gross_leverage`, `max_drawdown` |
| `portfolio_config` / `execution_config` / `tcost_config` / `slippage_config` | `{}` | Adapter hints |
| `model_config` / `walk_forward_config` / `scenario_config` | `{}` | Optional extensions |
| `checkpoint_dir` / `resume_from` | `None` | Pause/resume |
| `parallel` | `{}` | e.g. `grid` for sweeps |
| `enforce_pit` | `True` | Look-ahead invalidation |
| `reconciliation_tolerance` | `1e-4` | Capital identity |
| `meta` | `{}` | Free-form audit notes |

Aliases on load: `id` → `backtest_id`, `capital` → `initial_capital`. Universe may be a comma-separated string.

---

## Loaders

```python
from iqrp.app.backtesting.runner import BacktestRunConfig

cfg = BacktestRunConfig.from_yaml("iqrp/configs/backtesting/synthetic_demo.yaml")
cfg = BacktestRunConfig.from_dict({...})
cfg = BacktestRunConfig.from_omegaconf(omega_cfg)
cfg2 = cfg.with_updates(seed=99, output_dir="results")
root = cfg.results_root()  # Path(output_dir) / backtest_id
```

YAML prefers OmegaConf when available, else PyYAML `safe_load`.

---

## Reference configs

| File | Role |
|------|------|
| `synthetic_demo.yaml` | Smoke / demo against local synthetic parquet |
| `example_nifty50.yaml` | **Reference wiring only** — placeholder path; does **not** download data; **not** a profitability claim or recommendation |
| `default.yaml` | Hydra defaults for research `BacktestSettings` (engine), not the operational runner schema |

`example_nifty50.yaml` requires the user to replace `dataset_path` with a validated local file. Presence of that config is not evidence of NIFTY50 performance.

---

## Example YAML (operational)

```yaml
backtest_id: my_run
strategy_id: buy_and_hold
strategy_version: "1.0.0"
dataset_path: /path/to/your/ohlcv.parquet
dataset_id: user_supplied
dataset_version: "1.0.0"
adapter: parquet
universe: []
start: "2018-01-01"
end: "2023-12-31"
frequency: daily
initial_capital: 1000000.0
currency: INR
timezone: UTC
seed: 42
output_dir: results
commission_bps: 1.0
spread_bps: 2.0
slippage_bps: 1.0
risk_config:
  max_gross_leverage: 1.0
  max_drawdown: 1.0
strategy_params:
  mode: equal_weight
meta:
  purpose: research
```

---

## CLI overrides

`python -m iqrp.app.backtesting.run` merges flags onto the YAML/base config via `with_updates` (`--strategy`, `--dataset`, `--seed`, etc.). See [BacktestRunner](BacktestRunner.md).

---

## Critical rules

- Always set `dataset_path` to an existing local file.
- Always set `strategy_id` (+ version when ambiguous).
- Record `dataset_version`, checksum (via registry), `seed`, and code versions for reproducibility.
- Do not interpret reference YAML filenames (e.g. NIFTY50) as performance results.
