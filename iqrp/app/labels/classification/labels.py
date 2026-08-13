"""Classification labels."""

from __future__ import annotations

import numpy as np
import polars as pl

from iqrp.app.labels._utils import assign_quantile_buckets, with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import register_label
from iqrp.app.labels.config import LabelSettings


def _horizon() -> int:
    return LabelSettings.default().defaults.horizon


@register_label
class BinaryUp(Label):
    meta = LabelMeta(
        name="binary_up",
        version="1.0.0",
        description="1 if future return > threshold else 0",
        category="classification",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("binary_up",),
        parameters={
            "horizon": _horizon(),
            "threshold": LabelSettings.default().defaults.return_threshold,
        },
        dependencies=("future_return",),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        thr = float(self.meta.parameters.get("threshold", 0.0))
        fut = pl.col("close").shift(-h) / pl.col("close") - 1.0
        return with_open_time(frame, (fut > thr).cast(pl.Float64).alias("binary_up"))


@register_label
class BinaryDown(Label):
    meta = LabelMeta(
        name="binary_down",
        version="1.0.0",
        description="1 if future return < -threshold else 0",
        category="classification",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("binary_down",),
        parameters={
            "horizon": _horizon(),
            "threshold": LabelSettings.default().defaults.return_threshold,
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        thr = float(self.meta.parameters.get("threshold", 0.0))
        fut = pl.col("close").shift(-h) / pl.col("close") - 1.0
        return with_open_time(frame, (fut < -thr).cast(pl.Float64).alias("binary_down"))


@register_label
class MultiClassReturnBucket(Label):
    meta = LabelMeta(
        name="return_bucket",
        version="1.0.0",
        description="Quantile bucket of future return",
        category="classification",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("return_bucket",),
        parameters={
            "horizon": _horizon(),
            "quantiles": list(LabelSettings.default().defaults.bucket_quantiles),
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        qs = tuple(self.meta.parameters.get("quantiles", (0.25, 0.5, 0.75)))
        fut = (frame["close"].shift(-h) / frame["close"] - 1.0).to_numpy()
        buckets = assign_quantile_buckets(np.asarray(fut, dtype=np.float64), qs)
        return with_open_time(frame, pl.Series("return_bucket", buckets))


@register_label
class VolatilityBucket(Label):
    meta = LabelMeta(
        name="volatility_bucket",
        version="1.0.0",
        description="Quantile bucket of future realized volatility",
        category="classification",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("volatility_bucket",),
        parameters={"horizon": _horizon(), "window": 20, "quantiles": [0.33, 0.66]},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        w = int(self.meta.parameters.get("window", 20))
        qs = tuple(self.meta.parameters.get("quantiles", (0.33, 0.66)))
        vol = (
            frame.select(pl.col("close").pct_change().shift(-h).rolling_std(w).alias("v"))
            .to_series()
            .to_numpy()
        )
        return with_open_time(
            frame, pl.Series("volatility_bucket", assign_quantile_buckets(vol, qs))
        )


@register_label
class TrendBucket(Label):
    meta = LabelMeta(
        name="trend_bucket",
        version="1.0.0",
        description="Trend class from future return sign strength",
        category="classification",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("trend_bucket",),
        parameters={"horizon": _horizon(), "sideways": 0.002},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        side = float(self.meta.parameters.get("sideways", 0.002))
        fut = pl.col("close").shift(-h) / pl.col("close") - 1.0
        label = pl.when(fut > side).then(2.0).when(fut < -side).then(0.0).otherwise(1.0)
        return with_open_time(frame, label.alias("trend_bucket"))


@register_label
class RegimeClass(Label):
    meta = LabelMeta(
        name="regime_class",
        version="1.0.0",
        description="Future regime class (bull/sideways/bear) from trend bucket mapping",
        category="classification",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("regime_class",),
        dependencies=("trend_bucket",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        # Recompute trend-style class for independence
        h = int(self.meta.parameters.get("horizon", _horizon()))
        cfg = LabelSettings.default().regime
        fut = pl.col("close").shift(-h) / pl.col("close") - 1.0
        label = (
            pl.when(fut >= cfg.bull_threshold)
            .then(2.0)
            .when(fut <= cfg.bear_threshold)
            .then(0.0)
            .otherwise(1.0)
        )
        return with_open_time(frame, label.alias("regime_class"))


@register_label
class MarketStressClass(Label):
    meta = LabelMeta(
        name="market_stress_class",
        version="1.0.0",
        description="1 if future vol above stress quantile else 0",
        category="classification",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("market_stress_class",),
        parameters={
            "horizon": _horizon(),
            "window": 20,
            "quantile": LabelSettings.default().defaults.stress_vol_quantile,
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        w = int(self.meta.parameters.get("window", 20))
        q = float(self.meta.parameters.get("quantile", 0.9))
        vol = (
            frame.select(pl.col("close").pct_change().shift(-h).rolling_std(w).alias("v"))
            .to_series()
            .to_numpy()
        )
        finite = vol[np.isfinite(vol)]
        thr = float(np.quantile(finite, q)) if finite.size else float("inf")
        out = (vol >= thr).astype(float)
        out[~np.isfinite(vol)] = np.nan
        return with_open_time(frame, pl.Series("market_stress_class", out))
