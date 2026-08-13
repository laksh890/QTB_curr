"""Resume helpers for interrupted backtest runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from iqrp.app.backtesting.runner.checkpoint import apply_checkpoint, read_checkpoint
from iqrp.app.backtesting.runner.context import PipelineContext


def resume_timestamp(payload: dict[str, Any]) -> datetime | None:
    ctx = payload.get("context") or payload
    raw = ctx.get("current_time")
    if not raw:
        return None
    return datetime.fromisoformat(str(raw))


def restore_context(context: PipelineContext, checkpoint_file: str | Path) -> datetime | None:
    """Load checkpoint into context; return last event time for resume filtering."""
    payload = read_checkpoint(checkpoint_file)
    apply_checkpoint(context, payload)
    return resume_timestamp(payload)


__all__ = ["restore_context", "resume_timestamp"]
