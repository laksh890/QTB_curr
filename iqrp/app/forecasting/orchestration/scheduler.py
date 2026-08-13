"""Rolling retrain / checkpoint scheduling for online forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from iqrp.app.forecasting.config import ForecastingSettings


@dataclass
class ScheduleState:
    updates: int = 0
    last_retrain_at: int = 0
    last_checkpoint_at: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class ForecastScheduler:
    """Decides when to retrain, checkpoint, or stream-update a model."""

    def __init__(self, settings: ForecastingSettings | None = None) -> None:
        self.settings = settings or ForecastingSettings.default()
        self.state = ScheduleState()

    def should_retrain(self) -> bool:
        every = int(self.settings.online.rolling_retrain_every)
        if every <= 0:
            return False
        return (self.state.updates - self.state.last_retrain_at) >= every

    def should_checkpoint(self) -> bool:
        every = int(self.settings.online.checkpoint_every)
        if every <= 0:
            return False
        return (self.state.updates - self.state.last_checkpoint_at) >= every

    def on_update(
        self,
        model: Any,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> dict[str, Any]:
        self.state.updates += 1
        actions: dict[str, Any] = {"update": self.state.updates, "retrained": False, "checkpointed": False}
        if self.should_retrain() or not getattr(model, "is_fitted", False):
            model.fit(
                frame,
                feature_columns,
                target_column=target_column,
                regime_column=regime_column,
            )
            self.state.last_retrain_at = self.state.updates
            actions["retrained"] = True
        elif self.settings.online.warm_start:
            model.partial_fit(
                frame,
                feature_columns,
                target_column=target_column,
                regime_column=regime_column,
            )
        if self.should_checkpoint() and getattr(model, "is_fitted", False):
            model.checkpoint()
            self.state.last_checkpoint_at = self.state.updates
            actions["checkpointed"] = True
        self.state.history.append(actions)
        return actions

    def reset(self) -> None:
        self.state = ScheduleState()
