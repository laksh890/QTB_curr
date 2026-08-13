"""Automatically generated regime labels."""

from __future__ import annotations

import polars as pl

from iqrp.app.labels._utils import assign_quantile_buckets, with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import register_label
from iqrp.app.labels.config import LabelSettings


def _cfg() -> LabelSettings:
    return LabelSettings.default()


@register_label
class BullBearSideways(Label):
    meta = LabelMeta(
        name="bull_bear_sideways",
        version="1.0.0",
        description="0=bear, 1=sideways, 2=bull from trailing trend return",
        category="regime",
        prediction_horizon=0,
        required_inputs=("close",),
        output_columns=("bull_bear_sideways",),
        parameters={},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = _cfg().regime
        trend = pl.col("close") / pl.col("close").shift(cfg.trend_window) - 1.0
        label = (
            pl.when(trend >= cfg.bull_threshold)
            .then(2.0)
            .when(trend <= cfg.bear_threshold)
            .then(0.0)
            .otherwise(1.0)
        )
        return with_open_time(frame, label.alias("bull_bear_sideways"))


@register_label
class VolatilityRegime(Label):
    meta = LabelMeta(
        name="volatility_regime",
        version="1.0.0",
        description="Low/mid/high volatility regime from trailing realized vol",
        category="regime",
        prediction_horizon=0,
        required_inputs=("close",),
        output_columns=("volatility_regime",),
        parameters={},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = _cfg().regime
        vol = (
            frame.select(pl.col("close").pct_change().rolling_std(cfg.vol_window).alias("v"))
            .to_series()
            .to_numpy()
        )
        buckets = assign_quantile_buckets(vol, cfg.vol_quantiles)
        return with_open_time(frame, pl.Series("volatility_regime", buckets))


@register_label
class LiquidityRegime(Label):
    meta = LabelMeta(
        name="liquidity_regime",
        version="1.0.0",
        description="Liquidity regime from rolling volume quantiles",
        category="regime",
        prediction_horizon=0,
        required_inputs=("volume",),
        output_columns=("liquidity_regime",),
        parameters={},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = _cfg().regime
        liq = (
            frame.select(pl.col("volume").rolling_mean(cfg.liquidity_window).alias("v"))
            .to_series()
            .to_numpy()
        )
        return with_open_time(
            frame,
            pl.Series("liquidity_regime", assign_quantile_buckets(liq, cfg.liquidity_quantiles)),
        )


@register_label
class TrendRegime(Label):
    meta = LabelMeta(
        name="trend_regime",
        version="1.0.0",
        description="Trend regime from slope of close vs SMA",
        category="regime",
        prediction_horizon=0,
        required_inputs=("close",),
        output_columns=("trend_regime",),
        parameters={},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = _cfg().regime
        sma = pl.col("close").rolling_mean(cfg.trend_window)
        gap = (pl.col("close") - sma) / sma
        label = (
            pl.when(gap > cfg.sideways_threshold)
            .then(2.0)
            .when(gap < -cfg.sideways_threshold)
            .then(0.0)
            .otherwise(1.0)
        )
        return with_open_time(frame, label.alias("trend_regime"))


@register_label
class BullLabel(Label):
    meta = LabelMeta(
        name="bull",
        version="1.0.0",
        description="Binary bull regime flag",
        category="regime",
        prediction_horizon=0,
        required_inputs=("close",),
        output_columns=("bull",),
        dependencies=("bull_bear_sideways",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = _cfg().regime
        trend = pl.col("close") / pl.col("close").shift(cfg.trend_window) - 1.0
        return with_open_time(frame, (trend >= cfg.bull_threshold).cast(pl.Float64).alias("bull"))


@register_label
class BearLabel(Label):
    meta = LabelMeta(
        name="bear",
        version="1.0.0",
        description="Binary bear regime flag",
        category="regime",
        prediction_horizon=0,
        required_inputs=("close",),
        output_columns=("bear",),
        dependencies=("bull_bear_sideways",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = _cfg().regime
        trend = pl.col("close") / pl.col("close").shift(cfg.trend_window) - 1.0
        return with_open_time(frame, (trend <= cfg.bear_threshold).cast(pl.Float64).alias("bear"))


@register_label
class SidewaysLabel(Label):
    meta = LabelMeta(
        name="sideways",
        version="1.0.0",
        description="Binary sideways regime flag",
        category="regime",
        prediction_horizon=0,
        required_inputs=("close",),
        output_columns=("sideways",),
        dependencies=("bull_bear_sideways",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = _cfg().regime
        trend = pl.col("close") / pl.col("close").shift(cfg.trend_window) - 1.0
        mid = (trend < cfg.bull_threshold) & (trend > cfg.bear_threshold)
        return with_open_time(frame, mid.cast(pl.Float64).alias("sideways"))
