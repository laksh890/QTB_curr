"""Institutional human-readable and machine-readable backtest reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iqrp.app.backtesting.runner.result import OperationalBacktestResult


def _safe_pct(x: float) -> str:
    return f"{100.0 * float(x):.4f}%"


def build_report_payload(result: OperationalBacktestResult) -> dict[str, Any]:
    eq = list(result.equity_curve)
    start_eq = float(result.initial_capital or (eq[0] if eq else 0.0))
    end_eq = float(eq[-1]) if eq else start_eq
    total_ret = 0.0 if start_eq == 0 else end_eq / start_eq - 1.0
    perf = dict(result.performance or {})
    risk = dict(result.risk or {})
    return {
        "executive_summary": {
            "backtest_id": result.backtest_id,
            "status": result.status,
            "initial_capital": start_eq,
            "ending_equity": end_eq,
            "total_return": total_ret,
            "n_bars": len(eq),
            "n_orders": len(result.orders),
            "n_fills": len(result.fills),
            "note": (
                "Reference / research output only. Figures describe this run's "
                "simulated path under the stated assumptions; they are not a "
                "profitability claim or recommendation."
            ),
        },
        "performance": perf,
        "risk": risk,
        "drawdown": {
            "max_drawdown": risk.get("max_drawdown", perf.get("max_drawdown")),
            "end_drawdown": (
                (result.snapshots[-1] or {}).get("drawdown") if result.snapshots else None
            ),
        },
        "trading": {
            "orders": len(result.orders),
            "fills": len(result.fills),
            "trades": len(result.trades),
            "final_positions": result.positions_log[-1] if result.positions_log else [],
        },
        "execution": dict(result.execution or {}),
        "costs": {
            "fees_paid": (result.capital or {}).get("fees_paid"),
            "financing_paid": (result.capital or {}).get("financing_paid"),
        },
        "walk_forward": dict(result.walk_forward or {}),
        "scenarios": dict(result.scenarios or {}),
        "reproducibility": {
            "seed": result.seed,
            "strategy": {
                "id": (result.config or {}).get("strategy_id"),
                "version": (result.config or {}).get("strategy_version"),
            },
            "dataset": {
                "path": (result.config or {}).get("dataset_path"),
                "id": (result.config or {}).get("dataset_id"),
                "version": (result.config or {}).get("dataset_version"),
            },
            "backends": {
                "portfolio": (result.diagnostics or {}).get("portfolio_backend"),
                "execution": (result.diagnostics or {}).get("execution_backend"),
            },
        },
        "limitations": [
            "Simulated fills and costs are model-based approximations.",
            "Results are path-dependent on the supplied dataset and configuration.",
            "Reference strategies (e.g. buy_and_hold) are for pipeline validation only.",
            "No claim is made that historical simulated returns will persist.",
            "User must supply validated historical data; this platform does not download markets.",
        ],
        "reconciliation": dict(result.reconciliation or {}),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    ex = payload.get("executive_summary") or {}
    lines = [
        f"# Backtest Report — {ex.get('backtest_id', '')}",
        "",
        "## Executive Summary",
        "",
        f"- Status: `{ex.get('status')}`",
        f"- Initial capital: {ex.get('initial_capital')}",
        f"- Ending equity: {ex.get('ending_equity')}",
        f"- Total return (simulated): {_safe_pct(float(ex.get('total_return') or 0.0))}",
        f"- Bars: {ex.get('n_bars')} | Orders: {ex.get('n_orders')} | Fills: {ex.get('n_fills')}",
        "",
        f"> {ex.get('note', '')}",
        "",
        "## Performance",
        "",
        "```json",
        json.dumps(payload.get("performance") or {}, indent=2, default=str),
        "```",
        "",
        "## Risk",
        "",
        "```json",
        json.dumps(payload.get("risk") or {}, indent=2, default=str),
        "```",
        "",
        "## Drawdown",
        "",
        "```json",
        json.dumps(payload.get("drawdown") or {}, indent=2, default=str),
        "```",
        "",
        "## Trading",
        "",
        "```json",
        json.dumps(payload.get("trading") or {}, indent=2, default=str),
        "```",
        "",
        "## Execution",
        "",
        "```json",
        json.dumps(payload.get("execution") or {}, indent=2, default=str),
        "```",
        "",
        "## Costs",
        "",
        "```json",
        json.dumps(payload.get("costs") or {}, indent=2, default=str),
        "```",
        "",
        "## Walk-Forward",
        "",
        "```json",
        json.dumps(payload.get("walk_forward") or {}, indent=2, default=str),
        "```",
        "",
        "## Scenarios",
        "",
        "```json",
        json.dumps(payload.get("scenarios") or {}, indent=2, default=str),
        "```",
        "",
        "## Reproducibility",
        "",
        "```json",
        json.dumps(payload.get("reproducibility") or {}, indent=2, default=str),
        "```",
        "",
        "## Limitations",
        "",
    ]
    for item in payload.get("limitations") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Reconciliation",
            "",
            "```json",
            json.dumps(payload.get("reconciliation") or {}, indent=2, default=str),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    result: OperationalBacktestResult,
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir) / result.backtest_id / "reports"
    root.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(result)
    md_path = root / "report.md"
    json_path = root / "report.json"
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    result.reports = {
        "markdown": str(md_path),
        "json": str(json_path),
        "payload": payload,
    }
    return {"markdown": str(md_path), "json": str(json_path)}


__all__ = ["build_report_payload", "render_markdown", "write_reports"]
