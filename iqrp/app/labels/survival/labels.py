"""Survival-style labels: time-to-event / duration until barrier."""

from __future__ import annotations

import numpy as np
import polars as pl

from iqrp.app.labels._utils import atr, with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import register_label
from iqrp.app.labels.config import LabelSettings


@register_label
class TimeToUpperBarrier(Label):
    meta = LabelMeta(
        name="time_to_upper_barrier",
        version="1.0.0",
        description="Bars until price hits +ATR barrier (censored at horizon)",
        category="survival",
        prediction_horizon=LabelSettings.default().triple_barrier.horizon,
        required_inputs=("close", "high", "low"),
        output_columns=("time_to_upper_barrier", "upper_event"),
        parameters={
            "horizon": LabelSettings.default().triple_barrier.horizon,
            "atr_window": LabelSettings.default().triple_barrier.atr_window,
            "atr_multiplier": LabelSettings.default().triple_barrier.atr_multiplier,
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters["horizon"])
        aw = int(self.meta.parameters["atr_window"])
        mult = float(self.meta.parameters["atr_multiplier"])
        close = frame["close"].to_numpy().astype(np.float64)
        high = frame["high"].to_numpy().astype(np.float64)
        atr_s = atr(frame, aw).to_numpy().astype(np.float64)
        times = np.full(len(close), np.nan)
        events = np.full(len(close), np.nan)
        for i in range(len(close) - 1):
            if not np.isfinite(close[i]) or not np.isfinite(atr_s[i]):
                continue
            barrier = close[i] + mult * atr_s[i]
            hit = h
            event = 0.0
            end = min(len(close) - 1, i + h)
            for j in range(i + 1, end + 1):
                if high[j] >= barrier:
                    hit = j - i
                    event = 1.0
                    break
            times[i] = float(hit)
            events[i] = event
        return with_open_time(
            frame,
            pl.Series("time_to_upper_barrier", times),
            pl.Series("upper_event", events),
        )


@register_label
class TimeToLowerBarrier(Label):
    meta = LabelMeta(
        name="time_to_lower_barrier",
        version="1.0.0",
        description="Bars until price hits -ATR barrier (censored at horizon)",
        category="survival",
        prediction_horizon=LabelSettings.default().triple_barrier.horizon,
        required_inputs=("close", "high", "low"),
        output_columns=("time_to_lower_barrier", "lower_event"),
        parameters={
            "horizon": LabelSettings.default().triple_barrier.horizon,
            "atr_window": LabelSettings.default().triple_barrier.atr_window,
            "atr_multiplier": LabelSettings.default().triple_barrier.atr_multiplier,
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters["horizon"])
        aw = int(self.meta.parameters["atr_window"])
        mult = float(self.meta.parameters["atr_multiplier"])
        close = frame["close"].to_numpy().astype(np.float64)
        low = frame["low"].to_numpy().astype(np.float64)
        atr_s = atr(frame, aw).to_numpy().astype(np.float64)
        times = np.full(len(close), np.nan)
        events = np.full(len(close), np.nan)
        for i in range(len(close) - 1):
            if not np.isfinite(close[i]) or not np.isfinite(atr_s[i]):
                continue
            barrier = close[i] - mult * atr_s[i]
            hit = h
            event = 0.0
            end = min(len(close) - 1, i + h)
            for j in range(i + 1, end + 1):
                if low[j] <= barrier:
                    hit = j - i
                    event = 1.0
                    break
            times[i] = float(hit)
            events[i] = event
        return with_open_time(
            frame,
            pl.Series("time_to_lower_barrier", times),
            pl.Series("lower_event", events),
        )
