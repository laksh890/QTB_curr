"""Checkpoint serialization for pause / resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iqrp.app.backtesting.runner.context import PipelineContext


def checkpoint_path(root: str | Path, backtest_id: str) -> Path:
    return Path(root) / str(backtest_id) / "checkpoint.json"


def write_checkpoint(
    context: PipelineContext,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backtest_id": context.config.backtest_id,
        "seed": int(context.config.seed),
        "context": context.to_checkpoint(),
        "extra": dict(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def apply_checkpoint(context: PipelineContext, payload: dict[str, Any]) -> None:
    ctx_data = payload.get("context") if "context" in payload else payload
    context.load_checkpoint(dict(ctx_data or {}))


__all__ = ["apply_checkpoint", "checkpoint_path", "read_checkpoint", "write_checkpoint"]
