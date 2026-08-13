"""Shared helpers for label generation (Polars / NumPy)."""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl


def with_open_time(frame: pl.DataFrame, *exprs: pl.Expr | pl.Series) -> pl.DataFrame:
    cols: list[pl.Expr | pl.Series] = []
    if "open_time" in frame.columns:
        cols.append(pl.col("open_time"))
    cols.extend(exprs)
    return frame.select(cols)


def true_range(frame: pl.DataFrame) -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    ranges = [
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ]
    return pl.max_horizontal(*ranges)


def atr(frame: pl.DataFrame, window: int) -> pl.Series:
    return frame.select(true_range(frame).rolling_mean(window).alias("atr")).to_series()


def future_path_stats(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute forward return, MFE, MAE, and min drawdown over horizon."""
    n = len(close)
    fut_ret = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    drawdown = np.full(n, np.nan)
    for i in range(n - horizon):
        entry = close[i]
        if not np.isfinite(entry) or entry == 0:
            continue
        path_h = high[i + 1 : i + horizon + 1]
        path_l = low[i + 1 : i + horizon + 1]
        exit_px = close[i + horizon]
        fut_ret[i] = exit_px / entry - 1.0
        mfe[i] = float(np.max(path_h) / entry - 1.0)
        mae[i] = float(np.min(path_l) / entry - 1.0)
        # max adverse excursion as drawdown magnitude (negative)
        cummin = np.minimum.accumulate(path_l)
        drawdown[i] = float(np.min(cummin / entry - 1.0))
    return fut_ret, mfe, mae, drawdown


def assign_quantile_buckets(values: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    finite = values[np.isfinite(values)]
    out = np.full(len(values), np.nan)
    if finite.size < len(quantiles) + 2:
        return out
    edges = np.quantile(finite, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return out
    # digitize into 0..n buckets
    buckets = np.digitize(values, edges, right=True).astype(float)
    buckets[~np.isfinite(values)] = np.nan
    return cast(np.ndarray, buckets)
