"""Custom label API for user-defined prediction targets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.labels._utils import atr, with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import get_registry, register_label


def register_custom_label(
    name: str,
    *,
    compute_fn: Callable[[pl.DataFrame], pl.DataFrame],
    description: str,
    prediction_horizon: int,
    required_inputs: tuple[str, ...] = ("close",),
    output_columns: tuple[str, ...] | None = None,
    category: str = "custom",
    version: str = "1.0.0",
    parameters: dict[str, Any] | None = None,
    dependencies: tuple[str, ...] = (),
) -> type[Label]:
    """Dynamically create and register a custom label class."""

    outs = output_columns or (name,)

    class _Custom(Label):
        meta = LabelMeta(
            name=name,
            version=version,
            description=description,
            category=category,
            prediction_horizon=prediction_horizon,
            required_inputs=required_inputs,
            output_columns=outs,
            parameters=parameters or {},
            dependencies=dependencies,
        )

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            return compute_fn(frame)

    _Custom.__name__ = f"Custom_{name}"
    register_label(_Custom)
    return _Custom


def next_n_period_return(
    frame: pl.DataFrame, periods: int, *, name: str | None = None
) -> pl.DataFrame:
    """Next N-bar simple return."""
    col = name or f"return_{periods}"
    return with_open_time(
        frame, (pl.col("close").shift(-periods) / pl.col("close") - 1.0).alias(col)
    )


def probability_of_move(
    frame: pl.DataFrame,
    *,
    threshold: float,
    horizon: int,
    name: str = "prob_move",
) -> pl.DataFrame:
    """Binary indicator that |future return| exceeds ``threshold`` (softened to [0,1])."""
    fut = (pl.col("close").shift(-horizon) / pl.col("close") - 1.0).abs()
    # Smooth step around threshold
    soft = (1.0 / (1.0 + (-50.0 * (fut - threshold)).exp())).alias(name)
    return with_open_time(frame, soft)


def probability_of_atr_move(
    frame: pl.DataFrame,
    *,
    atr_multiple: float = 3.0,
    horizon: int = 12,
    atr_window: int = 14,
    name: str = "prob_atr_move",
) -> pl.DataFrame:
    """Probability-style label that |move| exceeds ``atr_multiple`` * ATR."""
    atr_s = atr(frame, atr_window).to_numpy()
    close = frame["close"].to_numpy()
    fut = np.full(len(close), np.nan)
    for i in range(len(close) - horizon):
        if not np.isfinite(close[i]) or not np.isfinite(atr_s[i]) or atr_s[i] == 0:
            continue
        move = abs(close[i + horizon] / close[i] - 1.0)
        thr = atr_multiple * atr_s[i] / close[i]
        # logistic around threshold
        fut[i] = 1.0 / (1.0 + np.exp(-50.0 * (move - thr)))
    return with_open_time(frame, pl.Series(name, fut))


def ensure_custom_examples_registered() -> None:
    """Register a few illustrative custom labels if not already present."""
    reg = get_registry()
    if "return_12" not in reg.list_names():

        def _ret12(frame: pl.DataFrame) -> pl.DataFrame:
            return next_n_period_return(frame, 12, name="return_12")

        register_custom_label(
            "return_12",
            compute_fn=_ret12,
            description="Next 12-bar return",
            prediction_horizon=12,
            output_columns=("return_12",),
        )

    if "prob_plus_2pct" not in reg.list_names():

        def _p2(frame: pl.DataFrame) -> pl.DataFrame:
            return probability_of_move(frame, threshold=0.02, horizon=12, name="prob_plus_2pct")

        register_custom_label(
            "prob_plus_2pct",
            compute_fn=_p2,
            description="Soft probability of |return| > 2%",
            prediction_horizon=12,
            output_columns=("prob_plus_2pct",),
        )

    if "prob_3atr_move" not in reg.list_names():

        def _p3(frame: pl.DataFrame) -> pl.DataFrame:
            return probability_of_atr_move(
                frame, atr_multiple=3.0, horizon=12, name="prob_3atr_move"
            )

        register_custom_label(
            "prob_3atr_move",
            compute_fn=_p3,
            description="Soft probability of 3 ATR move",
            prediction_horizon=12,
            required_inputs=("close", "high", "low"),
            output_columns=("prob_3atr_move",),
        )
