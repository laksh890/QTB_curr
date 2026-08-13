"""Future volatility estimator labels."""

from __future__ import annotations

import numpy as np
import polars as pl

from iqrp.app.labels._utils import with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import register_label
from iqrp.app.labels.config import LabelSettings


def _horizon() -> int:
    return LabelSettings.default().defaults.horizon


def _window() -> int:
    return LabelSettings.default().defaults.volatility_window


@register_label
class FutureRealizedVolatility(Label):
    meta = LabelMeta(
        name="future_realized_volatility",
        version="1.0.0",
        description="Future close-to-close realized volatility",
        category="volatility",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("future_realized_volatility",),
        parameters={"horizon": _horizon(), "window": _window()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters["horizon"])
        w = int(self.meta.parameters["window"])
        expr = pl.col("close").pct_change().shift(-h).rolling_std(w)
        return with_open_time(frame, expr.alias("future_realized_volatility"))


@register_label
class FutureParkinson(Label):
    meta = LabelMeta(
        name="future_parkinson",
        version="1.0.0",
        description="Future Parkinson volatility estimator",
        category="volatility",
        prediction_horizon=_horizon(),
        required_inputs=("high", "low"),
        output_columns=("future_parkinson",),
        parameters={"horizon": _horizon(), "window": _window()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters["horizon"])
        w = int(self.meta.parameters["window"])
        rs = (pl.col("high") / pl.col("low")).log().pow(2) / (4.0 * np.log(2.0))
        return with_open_time(frame, rs.shift(-h).rolling_mean(w).sqrt().alias("future_parkinson"))


@register_label
class FutureGarmanKlass(Label):
    meta = LabelMeta(
        name="future_garman_klass",
        version="1.0.0",
        description="Future Garman-Klass volatility estimator",
        category="volatility",
        prediction_horizon=_horizon(),
        required_inputs=("open", "high", "low", "close"),
        output_columns=("future_garman_klass",),
        parameters={"horizon": _horizon(), "window": _window()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters["horizon"])
        w = int(self.meta.parameters["window"])
        log_hl = (pl.col("high") / pl.col("low")).log().pow(2)
        log_co = (pl.col("close") / pl.col("open")).log().pow(2)
        gk = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
        return with_open_time(
            frame, gk.shift(-h).rolling_mean(w).clip(0, None).sqrt().alias("future_garman_klass")
        )


@register_label
class FutureYangZhang(Label):
    meta = LabelMeta(
        name="future_yang_zhang",
        version="1.0.0",
        description="Future Yang-Zhang volatility proxy",
        category="volatility",
        prediction_horizon=_horizon(),
        required_inputs=("open", "high", "low", "close"),
        output_columns=("future_yang_zhang",),
        parameters={"horizon": _horizon(), "window": _window()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters["horizon"])
        w = int(self.meta.parameters["window"])
        log_ho = (pl.col("high") / pl.col("open")).log()
        log_lo = (pl.col("low") / pl.col("open")).log()
        log_co = (pl.col("close") / pl.col("open")).log()
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        overnight = (pl.col("open") / pl.col("close").shift(1)).log().pow(2)
        yz = overnight + rs
        return with_open_time(
            frame, yz.shift(-h).rolling_mean(w).clip(0, None).sqrt().alias("future_yang_zhang")
        )


@register_label
class FutureEWMA(Label):
    meta = LabelMeta(
        name="future_ewma_volatility",
        version="1.0.0",
        description="Future EWMA volatility of returns",
        category="volatility",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("future_ewma_volatility",),
        parameters={"horizon": _horizon(), "span": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters["horizon"])
        span = int(self.meta.parameters.get("span", 20))
        ret = pl.col("close").pct_change()
        ewma = ret.pow(2).ewm_mean(span=span, adjust=False).sqrt()
        return with_open_time(frame, ewma.shift(-h).alias("future_ewma_volatility"))
