"""Streaming and batch inference helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.config import ForecastingSettings


@dataclass
class StreamingInference:
    """Buffered streaming predictor with optional online updates."""

    model: ForecastModel
    settings: ForecastingSettings | None = None
    buffer: deque[dict[str, Any]] = field(default_factory=deque)
    predictions: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.settings = self.settings or ForecastingSettings.default()
        maxlen = int(self.settings.online.stream_buffer)
        self.buffer = deque(self.buffer, maxlen=maxlen)

    def push(self, row: dict[str, Any]) -> float | None:
        self.buffer.append(row)
        if not self.model.is_fitted:
            return None
        frame = pl.DataFrame(list(self.buffer))
        pred = self.model.predict(frame)
        value = float(np.asarray(pred).reshape(-1)[-1])
        self.predictions.append(value)
        return value

    def forecast(self, *, horizon: int | None = None) -> Forecast:
        if not self.buffer:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("Stream buffer is empty", code="FC_EMPTY_BUFFER")
        frame = pl.DataFrame(list(self.buffer))
        h = horizon or self.settings.inference.default_horizon  # type: ignore[union-attr]
        return self.model.forecast(frame, horizon=h)


def batch_predict(
    model: ForecastModel,
    frame: pl.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """Memory-friendly batched ``predict`` over a large frame."""
    n = frame.height
    if n == 0:
        return np.array([], dtype=np.float64)
    bs = max(int(batch_size), 1)
    chunks: list[np.ndarray] = []
    for start in range(0, n, bs):
        sl = frame.slice(start, min(bs, n - start))
        chunks.append(np.asarray(model.predict(sl, feature_columns), dtype=np.float64).reshape(-1))
    return np.concatenate(chunks, axis=0)


def batch_forecast(
    model: ForecastModel,
    frame: pl.DataFrame,
    *,
    horizon: int = 1,
    feature_columns: list[str] | None = None,
) -> Forecast:
    return model.forecast(frame, horizon=horizon, feature_columns=feature_columns)
