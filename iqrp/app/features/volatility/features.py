"""Volatility features."""

from __future__ import annotations

import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


@register_feature
class ATR(Feature):
    meta = FeatureMeta(
        name="atr",
        version="1.0.0",
        description="Average True Range",
        category="volatility",
        required_columns=("high", "low", "close"),
        output_columns=("atr",),
        window=14,
        parameters={"window": 14},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        prev_close = pl.col("close").shift(1)
        tr = pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - prev_close).abs(),
            (pl.col("low") - prev_close).abs(),
        )
        return with_open_time(frame, tr.rolling_mean(w).alias("atr"))


@register_feature
class RollingStd(Feature):
    meta = FeatureMeta(
        name="rolling_std",
        version="1.0.0",
        description="Rolling standard deviation of returns",
        category="volatility",
        required_columns=("close",),
        output_columns=("rolling_std",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(
            frame, pl.col("close").pct_change().rolling_std(w).alias("rolling_std")
        )


@register_feature
class EwmaVolatility(Feature):
    meta = FeatureMeta(
        name="ewma_volatility",
        version="1.0.0",
        description="EWMA volatility of returns",
        category="volatility",
        required_columns=("close",),
        output_columns=("ewma_volatility",),
        window=20,
        parameters={"span": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        span = int(self.meta.parameters["span"])
        ret = pl.col("close").pct_change()
        evar = (ret * ret).ewm_mean(span=span, adjust=False)
        return with_open_time(frame, evar.sqrt().alias("ewma_volatility"))


@register_feature
class ParkinsonVolatility(Feature):
    meta = FeatureMeta(
        name="parkinson_volatility",
        version="1.0.0",
        description="Parkinson high-low volatility estimator",
        category="volatility",
        required_columns=("high", "low"),
        output_columns=("parkinson_volatility",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        # (1/(4*ln2)) * (ln(H/L))^2
        rs = (pl.col("high") / pl.col("low")).log().pow(2) / (4.0 * pl.lit(2.0).log())
        return with_open_time(frame, rs.rolling_mean(w).sqrt().alias("parkinson_volatility"))


@register_feature
class GarmanKlass(Feature):
    meta = FeatureMeta(
        name="garman_klass",
        version="1.0.0",
        description="Garman-Klass volatility estimator",
        category="volatility",
        required_columns=("open", "high", "low", "close"),
        output_columns=("garman_klass",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        log_hl = (pl.col("high") / pl.col("low")).log().pow(2)
        log_co = (pl.col("close") / pl.col("open")).log().pow(2)
        gk = 0.5 * log_hl - (2.0 * pl.lit(2.0).log() - 1.0) * log_co
        return with_open_time(frame, gk.rolling_mean(w).sqrt().alias("garman_klass"))


@register_feature
class YangZhang(Feature):
    meta = FeatureMeta(
        name="yang_zhang",
        version="1.0.0",
        description="Yang-Zhang volatility estimator (rolling)",
        category="volatility",
        required_columns=("open", "high", "low", "close"),
        output_columns=("yang_zhang",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        log_ho = (pl.col("high") / pl.col("open")).log()
        log_lo = (pl.col("low") / pl.col("open")).log()
        log_co = (pl.col("close") / pl.col("open")).log()
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        overnight = (pl.col("open") / pl.col("close").shift(1)).log()
        oc = log_co
        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        yz = (
            overnight.rolling_var(w) + k * oc.rolling_var(w) + (1.0 - k) * rs.rolling_mean(w)
        ).sqrt()
        return with_open_time(frame, yz.alias("yang_zhang"))


@register_feature
class HistoricalVolatility(Feature):
    meta = FeatureMeta(
        name="historical_volatility",
        version="1.0.0",
        description="Annualized historical volatility of log returns",
        category="volatility",
        required_columns=("close",),
        output_columns=("historical_volatility",),
        window=20,
        parameters={"window": 20, "periods_per_year": 365 * 24 * 60},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        ppy = float(self.meta.parameters["periods_per_year"])
        lr = (pl.col("close") / pl.col("close").shift(1)).log()
        return with_open_time(
            frame, (lr.rolling_std(w) * (ppy**0.5)).alias("historical_volatility")
        )


@register_feature
class RealizedVolatility(Feature):
    meta = FeatureMeta(
        name="realized_volatility",
        version="1.0.0",
        description="Sqrt of rolling sum of squared log returns",
        category="volatility",
        required_columns=("close",),
        output_columns=("realized_volatility",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        lr2 = (pl.col("close") / pl.col("close").shift(1)).log().pow(2)
        return with_open_time(frame, lr2.rolling_sum(w).sqrt().alias("realized_volatility"))


@register_feature
class VolatilityRegime(Feature):
    meta = FeatureMeta(
        name="volatility_regime",
        version="1.0.0",
        description="1 if short vol > long vol else 0",
        category="volatility",
        required_columns=("close",),
        output_columns=("volatility_regime",),
        dependencies=("rolling_std",),
        window=60,
        parameters={"short": 10, "long": 60},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        s = int(self.meta.parameters["short"])
        long_n = int(self.meta.parameters["long"])
        ret = pl.col("close").pct_change()
        short = ret.rolling_std(s)
        long = ret.rolling_std(long_n)
        return with_open_time(frame, (short > long).cast(pl.Float64).alias("volatility_regime"))


@register_feature
class VolatilityRatio(Feature):
    meta = FeatureMeta(
        name="volatility_ratio",
        version="1.0.0",
        description="Short / long rolling volatility",
        category="volatility",
        required_columns=("close",),
        output_columns=("volatility_ratio",),
        dependencies=("rolling_std",),
        window=60,
        parameters={"short": 10, "long": 60},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        s = int(self.meta.parameters["short"])
        long_n = int(self.meta.parameters["long"])
        ret = pl.col("close").pct_change()
        return with_open_time(
            frame,
            safe_div(ret.rolling_std(s), ret.rolling_std(long_n)).alias("volatility_ratio"),
        )
