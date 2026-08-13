"""Statistical features including Hurst and autocorrelation."""

from __future__ import annotations

import numpy as np
import polars as pl

from iqrp.app.features._polars_utils import safe_div, with_open_time
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.registry import register_feature


def _rolling_hurst(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(window - 1, len(values)):
        x = values[i - window + 1 : i + 1]
        if np.any(~np.isfinite(x)):
            continue
        y = np.cumsum(x - np.mean(x))
        r = np.max(y) - np.min(y)
        s = np.std(x, ddof=1)
        if s <= 0 or r <= 0:
            out[i] = 0.5
            continue
        out[i] = np.log(r / s) / np.log(window)
    return out


def _rolling_acf(values: np.ndarray, window: int, lag: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(window - 1, len(values)):
        x = values[i - window + 1 : i + 1]
        if lag >= len(x):
            continue
        a = x[lag:]
        b = x[:-lag]
        if a.std() == 0 or b.std() == 0:
            out[i] = 0.0
            continue
        out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


@register_feature
class RollingMean(Feature):
    meta = FeatureMeta(
        name="rolling_mean",
        version="1.0.0",
        description="Rolling mean of close",
        category="statistical",
        required_columns=("close",),
        output_columns=("rolling_mean",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(frame, pl.col("close").rolling_mean(w).alias("rolling_mean"))


@register_feature
class RollingMedian(Feature):
    meta = FeatureMeta(
        name="rolling_median",
        version="1.0.0",
        description="Rolling median of close",
        category="statistical",
        required_columns=("close",),
        output_columns=("rolling_median",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(frame, pl.col("close").rolling_median(w).alias("rolling_median"))


@register_feature
class RollingVariance(Feature):
    meta = FeatureMeta(
        name="rolling_variance",
        version="1.0.0",
        description="Rolling variance of close returns",
        category="statistical",
        required_columns=("close",),
        output_columns=("rolling_variance",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(
            frame, pl.col("close").pct_change().rolling_var(w).alias("rolling_variance")
        )


@register_feature
class RollingSkewness(Feature):
    meta = FeatureMeta(
        name="rolling_skewness",
        version="1.0.0",
        description="Rolling skewness of returns",
        category="statistical",
        required_columns=("close",),
        output_columns=("rolling_skewness",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        return with_open_time(
            frame, pl.col("close").pct_change().rolling_skew(w).alias("rolling_skewness")
        )


@register_feature
class RollingKurtosis(Feature):
    meta = FeatureMeta(
        name="rolling_kurtosis",
        version="1.0.0",
        description="Rolling kurtosis of returns",
        category="statistical",
        required_columns=("close",),
        output_columns=("rolling_kurtosis",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        rets = frame.select(pl.col("close").pct_change().fill_null(0.0)).to_series().to_numpy()
        arr = np.asarray(rets, dtype=np.float64)
        out = np.full(len(arr), np.nan, dtype=np.float64)
        for i in range(w - 1, len(arr)):
            x = arr[i - w + 1 : i + 1]
            if np.std(x) == 0:
                out[i] = 0.0
                continue
            # Excess kurtosis via moment formula
            m = np.mean(x)
            m4 = np.mean((x - m) ** 4)
            m2 = np.mean((x - m) ** 2)
            out[i] = (m4 / (m2**2) - 3.0) if m2 > 0 else 0.0
        return with_open_time(frame, pl.Series("rolling_kurtosis", out))


@register_feature
class RollingEntropy(Feature):
    meta = FeatureMeta(
        name="rolling_entropy",
        version="1.0.0",
        description="Approximate rolling entropy of return signs",
        category="statistical",
        required_columns=("close",),
        output_columns=("rolling_entropy",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        up = (pl.col("close").pct_change() > 0).cast(pl.Float64)
        p = up.rolling_mean(w).clip(1e-9, 1 - 1e-9)
        ent = -(p * p.log() + (1 - p) * (1 - p).log())
        return with_open_time(frame, ent.alias("rolling_entropy"))


@register_feature
class HurstExponent(Feature):
    meta = FeatureMeta(
        name="hurst_exponent",
        version="1.0.0",
        description="Rolling Hurst exponent via R/S",
        category="statistical",
        required_columns=("close",),
        output_columns=("hurst_exponent",),
        window=50,
        parameters={"window": 50},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        rets = frame.select(pl.col("close").pct_change().fill_null(0.0)).to_series().to_numpy()
        hurst = _rolling_hurst(np.asarray(rets, dtype=np.float64), w)
        return with_open_time(frame, pl.Series("hurst_exponent", hurst))


@register_feature
class Autocorrelation(Feature):
    meta = FeatureMeta(
        name="autocorrelation",
        version="1.0.0",
        description="Rolling lag-1 autocorrelation of returns",
        category="statistical",
        required_columns=("close",),
        output_columns=("autocorrelation",),
        window=30,
        parameters={"window": 30, "lag": 1},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        lag = int(self.meta.parameters["lag"])
        rets = frame.select(pl.col("close").pct_change().fill_null(0.0)).to_series().to_numpy()
        acf = _rolling_acf(np.asarray(rets, dtype=np.float64), w, lag)
        return with_open_time(frame, pl.Series("autocorrelation", acf))


@register_feature
class PartialAutocorrelation(Feature):
    meta = FeatureMeta(
        name="partial_autocorrelation",
        version="1.0.0",
        description="Rolling lag-1 PACF proxy (equals ACF at lag 1)",
        category="statistical",
        required_columns=("close",),
        output_columns=("partial_autocorrelation",),
        dependencies=("autocorrelation",),
        window=30,
        parameters={"window": 30},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        rets = frame.select(pl.col("close").pct_change().fill_null(0.0)).to_series().to_numpy()
        pacf = _rolling_acf(np.asarray(rets, dtype=np.float64), w, 1)
        return with_open_time(frame, pl.Series("partial_autocorrelation", pacf))


@register_feature
class ZScore(Feature):
    meta = FeatureMeta(
        name="zscore",
        version="1.0.0",
        description="Rolling z-score of close",
        category="statistical",
        required_columns=("close",),
        output_columns=("zscore",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        mu = pl.col("close").rolling_mean(w)
        sd = pl.col("close").rolling_std(w)
        return with_open_time(frame, safe_div(pl.col("close") - mu, sd).alias("zscore"))


@register_feature
class PercentileRank(Feature):
    meta = FeatureMeta(
        name="percentile_rank",
        version="1.0.0",
        description="Rolling percentile rank of close",
        category="statistical",
        required_columns=("close",),
        output_columns=("percentile_rank",),
        window=20,
        parameters={"window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        w = int(self.meta.parameters["window"])
        # Approximate via (close - min) / (max - min)
        mn = pl.col("close").rolling_min(w)
        mx = pl.col("close").rolling_max(w)
        return with_open_time(
            frame, safe_div(pl.col("close") - mn, mx - mn).alias("percentile_rank")
        )
