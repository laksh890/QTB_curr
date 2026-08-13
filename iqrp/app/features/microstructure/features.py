"""Microstructure features."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature
from iqrp.app.features.liquidity.features import _ensure_book_cols


@register_feature
class TradeImbalance(Feature):
    meta = FeatureMeta(
        name="trade_imbalance",
        version="1.0.0",
        description="Signed volume proxy from close changes",
        category="microstructure",
        required_columns=("close", "volume"),
        output_columns=("trade_imbalance",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        signed = pl.col("close").diff().sign().fill_null(0) * pl.col("volume")
        return with_open_time(frame, signed.rolling_sum(w).alias("trade_imbalance"))


@register_feature
class AmihudIlliquidity(Feature):
    meta = FeatureMeta(
        name="amihud_illiquidity",
        version="1.0.0",
        description="|return| / volume",
        category="microstructure",
        required_columns=("close", "volume"),
        output_columns=("amihud_illiquidity",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        illiq = safe_div(pl.col("close").pct_change().abs(), pl.col("volume"))
        return with_open_time(frame, illiq.rolling_mean(w).alias("amihud_illiquidity"))


@register_feature
class RollSpread(Feature):
    meta = FeatureMeta(
        name="roll_spread",
        version="1.0.0",
        description="Roll implied spread from return autocovariance",
        category="microstructure",
        required_columns=("close",),
        output_columns=("roll_spread",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        r = pl.col("close").pct_change()
        # cov(r_t, r_{t-1}) approx via rolling mean of product of demeaned
        mu = r.rolling_mean(w)
        cov = ((r - mu) * (r.shift(1) - mu)).rolling_mean(w)
        spread = (pl.max_horizontal(-cov, pl.lit(0.0)) * 4.0).sqrt()
        return with_open_time(frame, spread.alias("roll_spread"))


@register_feature
class Microprice(Feature):
    meta = FeatureMeta(
        name="microprice",
        version="1.0.0",
        description="Size-weighted mid price",
        category="microstructure",
        required_columns=("close", "volume"),
        output_columns=("microprice",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _ensure_book_cols(frame)
        mp = safe_div(
            pl.col("best_ask") * pl.col("bid_size") + pl.col("best_bid") * pl.col("ask_size"),
            pl.col("bid_size") + pl.col("ask_size"),
        )
        return with_open_time(f, mp.alias("microprice"))
