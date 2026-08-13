"""Volume features."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


@register_feature
class RelativeVolume(Feature):
    meta = FeatureMeta(
        name="relative_volume",
        version="1.0.0",
        description="Volume / rolling mean volume",
        category="volume",
        required_columns=("volume",),
        output_columns=("relative_volume",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(
            frame,
            safe_div(pl.col("volume"), pl.col("volume").rolling_mean(w)).alias("relative_volume"),
        )


@register_feature
class RollingVolume(Feature):
    meta = FeatureMeta(
        name="rolling_volume",
        version="1.0.0",
        description="Rolling mean of volume",
        category="volume",
        required_columns=("volume",),
        output_columns=("rolling_volume",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(frame, pl.col("volume").rolling_mean(w).alias("rolling_volume"))


@register_feature
class VolumeZScore(Feature):
    meta = FeatureMeta(
        name="volume_zscore",
        version="1.0.0",
        description="Z-score of volume",
        category="volume",
        required_columns=("volume",),
        output_columns=("volume_zscore",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        mu = pl.col("volume").rolling_mean(w)
        sd = pl.col("volume").rolling_std(w)
        return with_open_time(frame, safe_div(pl.col("volume") - mu, sd).alias("volume_zscore"))


@register_feature
class VolumeTrend(Feature):
    meta = FeatureMeta(
        name="volume_trend",
        version="1.0.0",
        description="Slope proxy: volume vs lag volume",
        category="volume",
        required_columns=("volume",),
        output_columns=("volume_trend",),
        window=5,
        parameters={"periods": 5},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        n = int(self.meta.parameters["periods"])
        return with_open_time(
            frame,
            (safe_div(pl.col("volume"), pl.col("volume").shift(n)) - 1.0).alias("volume_trend"),
        )


@register_feature
class VWAP(Feature):
    meta = FeatureMeta(
        name="vwap",
        version="1.0.0",
        description="Rolling VWAP using typical price",
        category="volume",
        required_columns=("high", "low", "close", "volume"),
        output_columns=("vwap",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        num = (tp * pl.col("volume")).rolling_sum(w)
        den = pl.col("volume").rolling_sum(w)
        return with_open_time(frame, safe_div(num, den).alias("vwap"))


@register_feature
class VWAPDistance(Feature):
    meta = FeatureMeta(
        name="vwap_distance",
        version="1.0.0",
        description="(close - vwap) / vwap",
        category="volume",
        required_columns=("high", "low", "close", "volume"),
        output_columns=("vwap_distance",),
        dependencies=("vwap",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        vwap = safe_div((tp * pl.col("volume")).rolling_sum(w), pl.col("volume").rolling_sum(w))
        return with_open_time(frame, safe_div(pl.col("close") - vwap, vwap).alias("vwap_distance"))


@register_feature
class OBV(Feature):
    meta = FeatureMeta(
        name="obv",
        version="1.0.0",
        description="On-Balance Volume",
        category="volume",
        required_columns=("close", "volume"),
        output_columns=("obv",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        direction = pl.col("close").diff().sign().fill_null(0)
        return with_open_time(frame, (direction * pl.col("volume")).cum_sum().alias("obv"))


@register_feature
class AccumulationDistribution(Feature):
    meta = FeatureMeta(
        name="accumulation_distribution",
        version="1.0.0",
        description="Accumulation/Distribution line",
        category="volume",
        required_columns=("high", "low", "close", "volume"),
        output_columns=("accumulation_distribution",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        mfm = safe_div(
            (pl.col("close") - pl.col("low") - (pl.col("high") - pl.col("close"))),
            pl.col("high") - pl.col("low"),
        )
        mfv = mfm * pl.col("volume")
        return with_open_time(frame, mfv.cum_sum().alias("accumulation_distribution"))


@register_feature
class ChaikinMoneyFlow(Feature):
    meta = FeatureMeta(
        name="chaikin_money_flow",
        version="1.0.0",
        description="Chaikin Money Flow",
        category="volume",
        required_columns=("high", "low", "close", "volume"),
        output_columns=("chaikin_money_flow",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        mfm = safe_div(
            (pl.col("close") - pl.col("low") - (pl.col("high") - pl.col("close"))),
            pl.col("high") - pl.col("low"),
        )
        mfv = mfm * pl.col("volume")
        return with_open_time(
            frame,
            safe_div(mfv.rolling_sum(w), pl.col("volume").rolling_sum(w)).alias(
                "chaikin_money_flow"
            ),
        )
