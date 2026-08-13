"""Trend features."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


@register_feature
class LogReturn(Feature):
    meta = FeatureMeta(
        name="log_return",
        version="1.0.0",
        description="Log return of close price",
        category="trend",
        required_columns=("close",),
        output_columns=("log_return",),
        window=1,
        parameters={"periods": 1},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        n = int(self.meta.parameters["periods"])
        return with_open_time(
            frame,
            (pl.col("close") / pl.col("close").shift(n)).log().alias("log_return"),
        )


@register_feature
class MultiPeriodReturn(Feature):
    meta = FeatureMeta(
        name="multi_period_return",
        version="1.0.0",
        description="Simple multi-period return of close",
        category="trend",
        required_columns=("close",),
        output_columns=("multi_period_return",),
        window=5,
        parameters={"periods": 5},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        n = int(self.meta.parameters["periods"])
        return with_open_time(
            frame,
            (pl.col("close") / pl.col("close").shift(n) - 1.0).alias("multi_period_return"),
        )


@register_feature
class EmaSlope(Feature):
    meta = FeatureMeta(
        name="ema_slope",
        version="1.0.0",
        description="First difference of EMA(close)",
        category="trend",
        required_columns=("close",),
        output_columns=("ema_slope",),
        window=20,
        parameters={"span": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        span = int(self.meta.parameters["span"])
        ema = pl.col("close").ewm_mean(span=span, adjust=False)
        return with_open_time(frame, (ema - ema.shift(1)).alias("ema_slope"))


@register_feature
class SmaSlope(Feature):
    meta = FeatureMeta(
        name="sma_slope",
        version="1.0.0",
        description="First difference of SMA(close)",
        category="trend",
        required_columns=("close",),
        output_columns=("sma_slope",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        sma = pl.col("close").rolling_mean(w)
        return with_open_time(frame, (sma - sma.shift(1)).alias("sma_slope"))


@register_feature
class LinearRegressionSlope(Feature):
    meta = FeatureMeta(
        name="linear_regression_slope",
        version="1.0.0",
        description="Rolling OLS slope of close vs time index",
        category="trend",
        required_columns=("close",),
        output_columns=("linear_regression_slope",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        # slope = cov(x,y)/var(x) with x = 0..w-1 => use rolling formulas
        idx = pl.int_range(0, pl.len()).cast(pl.Float64)
        x = idx
        y = pl.col("close")
        mean_x = x.rolling_mean(w)
        mean_y = y.rolling_mean(w)
        cov = (x * y).rolling_mean(w) - mean_x * mean_y
        var_x = (x * x).rolling_mean(w) - mean_x * mean_x
        slope = safe_div(cov, var_x)
        return with_open_time(frame, slope.alias("linear_regression_slope"))


@register_feature
class RollingTrend(Feature):
    meta = FeatureMeta(
        name="rolling_trend",
        version="1.0.0",
        description="Close relative to rolling SMA",
        category="trend",
        required_columns=("close",),
        output_columns=("rolling_trend",),
        dependencies=("sma_slope",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        sma = pl.col("close").rolling_mean(w)
        return with_open_time(frame, safe_div(pl.col("close") - sma, sma).alias("rolling_trend"))


@register_feature
class TrendStrength(Feature):
    meta = FeatureMeta(
        name="trend_strength",
        version="1.0.0",
        description="Absolute rolling return over window / rolling std",
        category="trend",
        required_columns=("close",),
        output_columns=("trend_strength",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        ret = pl.col("close") / pl.col("close").shift(w) - 1.0
        vol = pl.col("close").pct_change().rolling_std(w)
        return with_open_time(frame, safe_div(ret.abs(), vol).alias("trend_strength"))


@register_feature
class RollingHigh(Feature):
    meta = FeatureMeta(
        name="rolling_high",
        version="1.0.0",
        description="Rolling max of high",
        category="trend",
        required_columns=("high",),
        output_columns=("rolling_high",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(frame, pl.col("high").rolling_max(w).alias("rolling_high"))


@register_feature
class RollingLow(Feature):
    meta = FeatureMeta(
        name="rolling_low",
        version="1.0.0",
        description="Rolling min of low",
        category="trend",
        required_columns=("low",),
        output_columns=("rolling_low",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(frame, pl.col("low").rolling_min(w).alias("rolling_low"))


@register_feature
class DistanceFromHigh(Feature):
    meta = FeatureMeta(
        name="distance_from_high",
        version="1.0.0",
        description="(rolling_high - close) / rolling_high",
        category="trend",
        required_columns=("close", "high"),
        output_columns=("distance_from_high",),
        dependencies=("rolling_high",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        rh = pl.col("high").rolling_max(w)
        return with_open_time(frame, safe_div(rh - pl.col("close"), rh).alias("distance_from_high"))


@register_feature
class DistanceFromLow(Feature):
    meta = FeatureMeta(
        name="distance_from_low",
        version="1.0.0",
        description="(close - rolling_low) / rolling_low",
        category="trend",
        required_columns=("close", "low"),
        output_columns=("distance_from_low",),
        dependencies=("rolling_low",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        rl = pl.col("low").rolling_min(w)
        return with_open_time(frame, safe_div(pl.col("close") - rl, rl).alias("distance_from_low"))


@register_feature
class PriceAcceleration(Feature):
    meta = FeatureMeta(
        name="price_acceleration",
        version="1.0.0",
        description="Second difference of close",
        category="trend",
        required_columns=("close",),
        output_columns=("price_acceleration",),
        window=2,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        d1 = pl.col("close") - pl.col("close").shift(1)
        return with_open_time(frame, (d1 - d1.shift(1)).alias("price_acceleration"))


@register_feature
class PriceCurvature(Feature):
    meta = FeatureMeta(
        name="price_curvature",
        version="1.0.0",
        description="Normalized second difference of close",
        category="trend",
        required_columns=("close",),
        output_columns=("price_curvature",),
        dependencies=("price_acceleration",),
        window=2,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        d1 = pl.col("close") - pl.col("close").shift(1)
        accel = d1 - d1.shift(1)
        return with_open_time(frame, safe_div(accel, pl.col("close")).alias("price_curvature"))
