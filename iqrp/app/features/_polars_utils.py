"""Shared Polars helpers for feature implementations."""

from __future__ import annotations

import polars as pl


def require_ohlcv(frame: pl.DataFrame) -> None:
    needed = ("open", "high", "low", "close", "volume")
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError(
            f"OHLCV columns missing: {missing}",
            code="FEATURE_OHLCV_MISSING",
            details={"missing": list(missing)},
        )


def with_open_time(frame: pl.DataFrame, *exprs: pl.Expr | pl.Series) -> pl.DataFrame:
    cols: list[pl.Expr | pl.Series] = []
    if "open_time" in frame.columns:
        cols.append(pl.col("open_time"))
    cols.extend(exprs)
    return frame.select(cols)


def safe_div(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return pl.when(denom == 0).then(None).otherwise(numer / denom)
