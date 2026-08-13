"""Cross-asset features (multi-symbol frames)."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


@register_feature
class RelativeStrengthVsBenchmark(Feature):
    meta = FeatureMeta(
        name="relative_strength_vs_benchmark",
        version="1.0.0",
        description="close / benchmark_close - 1 when benchmark_close present",
        category="cross_asset",
        required_columns=("close",),
        output_columns=("relative_strength_vs_benchmark",),
        window=1,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "benchmark_close" in frame.columns:
            expr = safe_div(pl.col("close"), pl.col("benchmark_close")) - 1.0
        else:
            # Fallback: relative to rolling mean as local benchmark
            expr = safe_div(pl.col("close"), pl.col("close").rolling_mean(20)) - 1.0
        return with_open_time(frame, expr.alias("relative_strength_vs_benchmark"))


@register_feature
class SpreadToBenchmark(Feature):
    meta = FeatureMeta(
        name="spread_to_benchmark",
        version="1.0.0",
        description="close - benchmark_close",
        category="cross_asset",
        required_columns=("close",),
        output_columns=("spread_to_benchmark",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "benchmark_close" in frame.columns:
            expr = pl.col("close") - pl.col("benchmark_close")
        else:
            expr = pl.col("close") - pl.col("close").rolling_mean(20)
        return with_open_time(frame, expr.alias("spread_to_benchmark"))


@register_feature
class BetaToBenchmark(Feature):
    meta = FeatureMeta(
        name="beta_to_benchmark",
        version="1.0.0",
        description="Rolling beta of asset returns vs benchmark returns",
        category="cross_asset",
        required_columns=("close",),
        output_columns=("beta_to_benchmark",),
        window=30,
        parameters={"window": 30},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        r = pl.col("close").pct_change()
        if "benchmark_close" in frame.columns:
            b = pl.col("benchmark_close").pct_change()
        else:
            b = pl.col("close").rolling_mean(20).pct_change()
        cov = (r * b).rolling_mean(w) - r.rolling_mean(w) * b.rolling_mean(w)
        var = b.rolling_var(w)
        return with_open_time(frame, safe_div(cov, var).alias("beta_to_benchmark"))
