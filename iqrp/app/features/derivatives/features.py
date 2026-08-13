"""Derivatives features (funding, OI, liquidations, basis)."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


def _with_defaults(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    if "funding_rate" not in out.columns:
        out = out.with_columns(pl.lit(0.0).alias("funding_rate"))
    if "open_interest" not in out.columns:
        out = out.with_columns(pl.col("volume").alias("open_interest"))
    if "long_short_ratio" not in out.columns:
        out = out.with_columns(pl.lit(1.0).alias("long_short_ratio"))
    if "liquidation_count" not in out.columns:
        out = out.with_columns(pl.lit(0.0).alias("liquidation_count"))
    if "liquidation_volume" not in out.columns:
        out = out.with_columns(pl.lit(0.0).alias("liquidation_volume"))
    if "mark_price" not in out.columns:
        out = out.with_columns(pl.col("close").alias("mark_price"))
    if "index_price" not in out.columns:
        out = out.with_columns(pl.col("close").alias("index_price"))
    return out


@register_feature
class FundingRateFeature(Feature):
    meta = FeatureMeta(
        name="funding_rate",
        version="1.0.0",
        description="Perpetual funding rate (passthrough/defaulted)",
        category="derivatives",
        required_columns=("close",),
        output_columns=("funding_rate_feat",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(f, pl.col("funding_rate").alias("funding_rate_feat"))


@register_feature
class FundingMomentum(Feature):
    meta = FeatureMeta(
        name="funding_momentum",
        version="1.0.0",
        description="Change in funding rate",
        category="derivatives",
        required_columns=("close",),
        output_columns=("funding_momentum",),
        dependencies=("funding_rate",),
        window=1,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(
            f, (pl.col("funding_rate") - pl.col("funding_rate").shift(1)).alias("funding_momentum")
        )


@register_feature
class FundingMeanReversion(Feature):
    meta = FeatureMeta(
        name="funding_mean_reversion",
        version="1.0.0",
        description="Funding z-score (mean reversion signal input)",
        category="derivatives",
        required_columns=("close",),
        output_columns=("funding_mean_reversion",),
        dependencies=("funding_rate",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        w = int(self.meta.parameters["window"])
        mu = pl.col("funding_rate").rolling_mean(w)
        sd = pl.col("funding_rate").rolling_std(w)
        return with_open_time(
            f, safe_div(pl.col("funding_rate") - mu, sd).alias("funding_mean_reversion")
        )


@register_feature
class OpenInterestFeature(Feature):
    meta = FeatureMeta(
        name="open_interest",
        version="1.0.0",
        description="Open interest level",
        category="derivatives",
        required_columns=("volume",),
        output_columns=("open_interest_feat",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(f, pl.col("open_interest").alias("open_interest_feat"))


@register_feature
class OIMomentum(Feature):
    meta = FeatureMeta(
        name="oi_momentum",
        version="1.0.0",
        description="Open interest change",
        category="derivatives",
        required_columns=("volume",),
        output_columns=("oi_momentum",),
        dependencies=("open_interest",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(
            f, (pl.col("open_interest") - pl.col("open_interest").shift(1)).alias("oi_momentum")
        )


@register_feature
class LongShortRatioFeature(Feature):
    meta = FeatureMeta(
        name="long_short_ratio",
        version="1.0.0",
        description="Long/short ratio",
        category="derivatives",
        required_columns=("close",),
        output_columns=("long_short_ratio_feat",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(f, pl.col("long_short_ratio").alias("long_short_ratio_feat"))


@register_feature
class LiquidationCountFeature(Feature):
    meta = FeatureMeta(
        name="liquidation_count",
        version="1.0.0",
        description="Liquidation event count",
        category="derivatives",
        required_columns=("close",),
        output_columns=("liquidation_count_feat",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(f, pl.col("liquidation_count").alias("liquidation_count_feat"))


@register_feature
class LiquidationVolumeFeature(Feature):
    meta = FeatureMeta(
        name="liquidation_volume",
        version="1.0.0",
        description="Liquidation volume",
        category="derivatives",
        required_columns=("close",),
        output_columns=("liquidation_volume_feat",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(f, pl.col("liquidation_volume").alias("liquidation_volume_feat"))


@register_feature
class Basis(Feature):
    meta = FeatureMeta(
        name="basis",
        version="1.0.0",
        description="(mark - index) / index",
        category="derivatives",
        required_columns=("close",),
        output_columns=("basis",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        f = _with_defaults(frame)
        return with_open_time(
            f,
            safe_div(pl.col("mark_price") - pl.col("index_price"), pl.col("index_price")).alias(
                "basis"
            ),
        )
