"""Automatic retraining policies and warm-start recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from iqrp.app.forecasting.intelligence.config import RetrainConfig
from iqrp.app.forecasting.intelligence.drift import DriftReport


@dataclass
class RetrainDecision:
    should_retrain: bool
    reason: str
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "reason": self.reason,
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }


def decide_retrain(
    *,
    n_updates: int,
    config: RetrainConfig,
    drift: DriftReport | None = None,
    performance_degraded: bool = False,
) -> RetrainDecision:
    if config.mode == "none":
        return RetrainDecision(False, "disabled", config.mode)
    if (
        config.mode == "scheduled"
        and n_updates > 0
        and n_updates % max(config.schedule_every, 1) == 0
    ):
        return RetrainDecision(True, "schedule", config.mode, {"n_updates": n_updates})
    if config.mode == "drift" and drift is not None and drift.triggered:
        return RetrainDecision(True, "drift:" + ",".join(drift.reasons), config.mode)
    if config.mode == "performance" and performance_degraded:
        return RetrainDecision(True, "performance", config.mode)
    if config.mode == "rolling":
        return RetrainDecision(True, "rolling", config.mode, {"window": config.window})
    # performance mode also reacts to drift
    if config.mode == "performance" and drift is not None and drift.triggered:
        return RetrainDecision(True, "drift:" + ",".join(drift.reasons), config.mode)
    return RetrainDecision(False, "ok", config.mode)


def retrain_model(
    model: Any,
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    config: RetrainConfig,
    warm_start: bool | None = None,
) -> Any:
    use_warm = config.warm_start if warm_start is None else warm_start
    window = max(int(config.window), 16)
    data = frame[-window:] if frame.height > window else frame
    if use_warm and hasattr(model, "partial_fit") and getattr(model, "is_fitted", False):
        try:
            return model.partial_fit(
                data, feature_columns=feature_columns, target_column=target_column
            )
        except Exception:
            pass
    return model.fit(data, feature_columns=feature_columns, target_column=target_column)


def checkpoint_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "checkpoint"):
        return model.checkpoint()
    if hasattr(model, "export_state"):
        return model.export_state()
    return {}


def restore_checkpoint(model: Any, payload: dict[str, Any]) -> Any:
    if hasattr(model, "restore_checkpoint"):
        return model.restore_checkpoint(payload)
    if hasattr(model, "import_state"):
        model.import_state(payload)
        return model
    return model
