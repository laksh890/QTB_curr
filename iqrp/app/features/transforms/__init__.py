"""Generic Polars transforms for feature matrices."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl
from scipy import stats  # type: ignore[import-untyped]


def lag(frame: pl.DataFrame, columns: Iterable[str], periods: int = 1) -> pl.DataFrame:
    exprs = [pl.col(c).shift(periods).alias(f"{c}_lag{periods}") for c in columns]
    return frame.with_columns(exprs)


def difference(frame: pl.DataFrame, columns: Iterable[str], periods: int = 1) -> pl.DataFrame:
    exprs = [pl.col(c).diff(periods).alias(f"{c}_diff{periods}") for c in columns]
    return frame.with_columns(exprs)


def percentage_change(
    frame: pl.DataFrame, columns: Iterable[str], periods: int = 1
) -> pl.DataFrame:
    exprs = [pl.col(c).pct_change(periods).alias(f"{c}_pct{periods}") for c in columns]
    return frame.with_columns(exprs)


def rolling_window(
    frame: pl.DataFrame, columns: Iterable[str], window: int, *, agg: str = "mean"
) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for c in columns:
        col = pl.col(c)
        if agg == "mean":
            exprs.append(col.rolling_mean(window).alias(f"{c}_rollmean{window}"))
        elif agg == "std":
            exprs.append(col.rolling_std(window).alias(f"{c}_rollstd{window}"))
        elif agg == "sum":
            exprs.append(col.rolling_sum(window).alias(f"{c}_rollsum{window}"))
        else:
            raise ValueError(f"Unsupported agg: {agg}")
    return frame.with_columns(exprs)


def expanding_window(
    frame: pl.DataFrame, columns: Iterable[str], *, agg: str = "mean"
) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for c in columns:
        if agg == "mean":
            exprs.append(
                (pl.col(c).cum_sum() / pl.int_range(1, pl.len() + 1)).alias(f"{c}_expmean")
            )
        elif agg == "sum":
            exprs.append(pl.col(c).cum_sum().alias(f"{c}_expsum"))
        else:
            raise ValueError(f"Unsupported agg: {agg}")
    return frame.with_columns(exprs)


def normalize_minmax(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    exprs = []
    for c in columns:
        mn = pl.col(c).min()
        mx = pl.col(c).max()
        exprs.append(
            pl.when(mx == mn).then(0.0).otherwise((pl.col(c) - mn) / (mx - mn)).alias(f"{c}_norm")
        )
    return frame.with_columns(exprs)


def standardize(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    exprs = []
    for c in columns:
        mu = pl.col(c).mean()
        sd = pl.col(c).std()
        exprs.append(
            pl.when((sd == 0) | sd.is_null())
            .then(0.0)
            .otherwise((pl.col(c) - mu) / sd)
            .alias(f"{c}_z")
        )
    return frame.with_columns(exprs)


def winsorize(
    frame: pl.DataFrame, columns: Iterable[str], *, lower: float = 0.01, upper: float = 0.99
) -> pl.DataFrame:
    exprs = []
    for c in columns:
        lo = pl.col(c).quantile(lower)
        hi = pl.col(c).quantile(upper)
        exprs.append(pl.col(c).clip(lo, hi).alias(f"{c}_wins"))
    return frame.with_columns(exprs)


def log_transform(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    exprs = [pl.col(c).log().alias(f"{c}_log") for c in columns]
    return frame.with_columns(exprs)


def box_cox_transform(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    series = frame[column].to_numpy()
    positive = np.asarray(series, dtype=np.float64)
    min_v = np.nanmin(positive) if len(positive) else 0.0
    shifted = positive - min_v + 1e-6 if min_v <= 0 else positive
    mask = np.isfinite(shifted) & (shifted > 0)
    out = np.full_like(shifted, np.nan)
    if int(mask.sum()) > 5:
        transformed, _lambda = stats.boxcox(shifted[mask])
        out[mask] = transformed
    return frame.with_columns(pl.Series(f"{column}_boxcox", out))
