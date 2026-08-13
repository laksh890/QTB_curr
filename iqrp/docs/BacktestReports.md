# Backtest Reports

Institutional human-readable and machine-readable reports for operational Phase 13 runs.

---

## Purpose

`write_reports` builds a structured payload (executive summary, performance, risk, trading, execution, costs, walk-forward, scenarios, reproducibility, limitations, reconciliation) and writes:

- `{output_dir}/{backtest_id}/reports/report.md`
- `{output_dir}/{backtest_id}/reports/report.json`

**Module:** `iqrp.app.backtesting.runner.reports`  
**Related:** [ResultStorage](ResultStorage.md) · [BacktestRunner](BacktestRunner.md) · [UserGuide](UserGuide.md)

---

## Payload sections

| Section | Contents |
|---------|----------|
| `executive_summary` | backtest_id, status, initial/ending equity, total return (simulated), bar/order/fill counts, disclaimer note |
| `performance` | From `summarize_returns` / Sharpe when available |
| `risk` | max drawdown, leverage/vol/VaR fields from context |
| `drawdown` | max + end snapshot drawdown |
| `trading` | order/fill/trade counts, final positions |
| `execution` | backend, order counts |
| `costs` | fees_paid, financing_paid |
| `walk_forward` / `scenarios` | Optional extension summaries |
| `reproducibility` | seed, strategy id/version, dataset path/id/version, backends |
| `limitations` | Explicit model/data caveats |
| `reconciliation` | Capital identity result |

Executive note (verbatim intent):

> Reference / research output only. Figures describe this run's simulated path under the stated assumptions; they are not a profitability claim or recommendation.

Limitations include: simulated fills/costs are approximations; path dependence on supplied data/config; reference strategies are for pipeline validation; no claim that historical simulated returns persist; user must supply validated data (platform does not download markets).

---

## API

```python
from iqrp.app.backtesting.runner.reports import (
    build_report_payload,
    render_markdown,
    write_reports,
)

payload = build_report_payload(result)
md = render_markdown(payload)
paths = write_reports(result, output_dir="results")
# paths["markdown"], paths["json"]
```

`BacktestRunner.report()` returns the markdown path (preferred) or JSON path.

---

## Reading a report

```bash
less results/synthetic_demo/reports/report.md
python -c "import json; print(json.load(open('results/synthetic_demo/reports/report.json'))['executive_summary'])"
```

Interpret **total return** and Sharpe as properties of this simulated path only. Reference configs such as `example_nifty50.yaml` do not imply NIFTY50 profitability.

---

## Critical rules

- Always surface limitations and reproducibility blocks.
- Never present report figures as live trading expectations.
- Keep reports next to persisted ledgers for audit.
