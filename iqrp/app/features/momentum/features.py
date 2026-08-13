"""Momentum features."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


@register_feature
class Momentum(Feature):
    meta = FeatureMeta(
        name="momentum",
        version="1.0.0",
        description="Close - close.n periods ago",
        category="momentum",
        required_columns=("close",),
        output_columns=("momentum",),
        window=10,
        parameters={"periods": 10},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        n = int(self.meta.parameters["periods"])
        return with_open_time(frame, (pl.col("close") - pl.col("close").shift(n)).alias("momentum"))


@register_feature
class ROC(Feature):
    meta = FeatureMeta(
        name="roc",
        version="1.0.0",
        description="Rate of change percent",
        category="momentum",
        required_columns=("close",),
        output_columns=("roc",),
        window=10,
        parameters={"periods": 10},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        n = int(self.meta.parameters["periods"])
        return with_open_time(
            frame,
            ((pl.col("close") / pl.col("close").shift(n) - 1.0) * 100.0).alias("roc"),
        )


@register_feature
class RSI(Feature):
    meta = FeatureMeta(
        name="rsi",
        version="1.0.0",
        description="Relative Strength Index",
        category="momentum",
        required_columns=("close",),
        output_columns=("rsi",),
        window=14,
        parameters={"window": 14},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        delta = pl.col("close").diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0.0).rolling_mean(w)
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0).rolling_mean(w)
        rs = safe_div(gain, loss)
        rsi = 100.0 - safe_div(pl.lit(100.0), 1.0 + rs)
        return with_open_time(frame, rsi.alias("rsi"))


@register_feature
class StochasticOscillator(Feature):
    meta = FeatureMeta(
        name="stochastic_oscillator",
        version="1.0.0",
        description="Stochastic %K",
        category="momentum",
        required_columns=("high", "low", "close"),
        output_columns=("stochastic_k",),
        window=14,
        parameters={"window": 14},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        lowest = pl.col("low").rolling_min(w)
        highest = pl.col("high").rolling_max(w)
        k = safe_div(pl.col("close") - lowest, highest - lowest) * 100.0
        return with_open_time(frame, k.alias("stochastic_k"))


@register_feature
class WilliamsR(Feature):
    meta = FeatureMeta(
        name="williams_r",
        version="1.0.0",
        description="Williams %R",
        category="momentum",
        required_columns=("high", "low", "close"),
        output_columns=("williams_r",),
        window=14,
        parameters={"window": 14},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        highest = pl.col("high").rolling_max(w)
        lowest = pl.col("low").rolling_min(w)
        wr = safe_div(highest - pl.col("close"), highest - lowest) * -100.0
        return with_open_time(frame, wr.alias("williams_r"))


@register_feature
class CCI(Feature):
    meta = FeatureMeta(
        name="cci",
        version="1.0.0",
        description="Commodity Channel Index",
        category="momentum",
        required_columns=("high", "low", "close"),
        output_columns=("cci",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        sma = tp.rolling_mean(w)
        mad = (tp - sma).abs().rolling_mean(w)
        cci = safe_div(tp - sma, 0.015 * mad)
        return with_open_time(frame, cci.alias("cci"))


@register_feature
class MACDComponents(Feature):
    meta = FeatureMeta(
        name="macd_components",
        version="1.0.0",
        description="MACD line, signal, and histogram",
        category="momentum",
        required_columns=("close",),
        output_columns=("macd", "macd_signal", "macd_hist"),
        window=26,
        parameters={"fast": 12, "slow": 26, "signal": 9},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        fast = int(self.meta.parameters["fast"])
        slow = int(self.meta.parameters["slow"])
        signal = int(self.meta.parameters["signal"])
        ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
        ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
        macd = ema_fast - ema_slow
        sig = macd.ewm_mean(span=signal, adjust=False)
        return with_open_time(
            frame,
            macd.alias("macd"),
            sig.alias("macd_signal"),
            (macd - sig).alias("macd_hist"),
        )


@register_feature
class MomentumPersistence(Feature):
    meta = FeatureMeta(
        name="momentum_persistence",
        version="1.0.0",
        description="Fraction of positive returns in window",
        category="momentum",
        required_columns=("close",),
        output_columns=("momentum_persistence",),
        dependencies=("momentum",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        up = (pl.col("close").pct_change() > 0).cast(pl.Float64)
        return with_open_time(frame, up.rolling_mean(w).alias("momentum_persistence"))


@register_feature
class MomentumDecay(Feature):
    meta = FeatureMeta(
        name="momentum_decay",
        version="1.0.0",
        description="Short momentum divided by long momentum",
        category="momentum",
        required_columns=("close",),
        output_columns=("momentum_decay",),
        dependencies=("momentum",),
        window=20,
        parameters={"short": 5, "long": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        s = int(self.meta.parameters["short"])
        long_n = int(self.meta.parameters["long"])
        short_m = pl.col("close") - pl.col("close").shift(s)
        long_m = pl.col("close") - pl.col("close").shift(long_n)
        return with_open_time(frame, safe_div(short_m, long_m).alias("momentum_decay"))
