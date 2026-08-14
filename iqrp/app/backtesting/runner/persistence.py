"""Persist operational backtest artifacts under results/{backtest_id}/."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.runner.result import OperationalBacktestResult

ARTIFACT_DIRS = (
    "configuration",
    "data",
    "orders",
    "fills",
    "trades",
    "positions",
    "portfolio",
    "risk",
    "performance",
    "execution",
    "walk_forward",
    "scenarios",
    "diagnostics",
    "reports",
)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _write_table(path: Path, rows: list[Mapping[str, Any]]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _write_json(path.with_suffix(".json"), [])
        return path.with_suffix(".json")
    df = pd.DataFrame(list(rows))
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path.with_suffix(".csv")


def persist_result(
    result: OperationalBacktestResult,
    output_dir: str | Path,
) -> Path:
    root = Path(output_dir) / result.backtest_id
    for name in ARTIFACT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)

    _write_json(root / "configuration" / "config.json", result.config)
    _write_json(root / "data" / "summary.json", result.diagnostics.get("data_detail", {}))
    _write_table(root / "orders" / "orders", result.orders)
    _write_table(root / "fills" / "fills", result.fills)
    _write_table(root / "trades" / "trades", result.trades)
    _write_table(root / "positions" / "positions", result.positions_log)
    _write_table(root / "portfolio" / "snapshots", result.snapshots)
    _write_json(
        root / "portfolio" / "equity_curve.json",
        {"timestamps": result.timestamps, "equity": result.equity_curve, "returns": result.returns},
    )
    _write_json(root / "risk" / "risk.json", result.risk)
    _write_json(root / "performance" / "performance.json", result.performance)
    _write_json(root / "execution" / "execution.json", result.execution)
    _write_json(root / "walk_forward" / "walk_forward.json", result.walk_forward)
    _write_json(root / "scenarios" / "scenarios.json", result.scenarios)
    _write_json(root / "diagnostics" / "diagnostics.json", result.diagnostics)
    _write_json(root / "diagnostics" / "reconciliation.json", result.reconciliation)
    _write_json(root / "reports" / "result.json", result.to_dict())
    _write_json(root / "capital.json", result.capital)
    return root


__all__ = ["ARTIFACT_DIRS", "persist_result"]
