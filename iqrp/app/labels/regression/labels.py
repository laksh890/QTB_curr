"""Regression labels: future returns, vol, drawdown, MFE/MAE, etc."""

from __future__ import annotations

import polars as pl

from iqrp.app.labels._utils import atr, future_path_stats, with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import register_label
from iqrp.app.labels.config import LabelSettings


def _horizon() -> int:
    return LabelSettings.default().defaults.horizon


@register_label
class FutureReturn(Label):
    meta = LabelMeta(
        name="future_return",
        version="1.0.0",
        description="Close-to-close simple return over prediction horizon",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("future_return",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        return with_open_time(
            frame, (pl.col("close").shift(-h) / pl.col("close") - 1.0).alias("future_return")
        )


@register_label
class FutureLogReturn(Label):
    meta = LabelMeta(
        name="future_log_return",
        version="1.0.0",
        description="Log return over prediction horizon",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("future_log_return",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        return with_open_time(
            frame,
            (pl.col("close").shift(-h).log() - pl.col("close").log()).alias("future_log_return"),
        )


@register_label
class FutureVolatility(Label):
    meta = LabelMeta(
        name="future_volatility",
        version="1.0.0",
        description="Realized volatility of returns over next horizon window",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close",),
        output_columns=("future_volatility",),
        parameters={
            "horizon": _horizon(),
            "window": LabelSettings.default().defaults.volatility_window,
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        w = int(self.meta.parameters.get("window", 20))
        ret = pl.col("close").pct_change()
        return with_open_time(frame, ret.shift(-h).rolling_std(w).alias("future_volatility"))


@register_label
class FutureATR(Label):
    meta = LabelMeta(
        name="future_atr",
        version="1.0.0",
        description="ATR measured at t+horizon",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("high", "low", "close"),
        output_columns=("future_atr",),
        parameters={"horizon": _horizon(), "window": LabelSettings.default().defaults.atr_window},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        w = int(self.meta.parameters.get("window", 14))
        series = atr(frame, w).shift(-h)
        return with_open_time(frame, pl.Series("future_atr", series))


@register_label
class FutureDrawdown(Label):
    meta = LabelMeta(
        name="future_drawdown",
        version="1.0.0",
        description="Maximum forward drawdown over horizon",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close", "high", "low"),
        output_columns=("future_drawdown",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        close = frame["close"].to_numpy()
        high = frame["high"].to_numpy()
        low = frame["low"].to_numpy()
        _, _, _, dd = future_path_stats(close, high, low, horizon=h)
        return with_open_time(frame, pl.Series("future_drawdown", dd))


@register_label
class FutureMFE(Label):
    meta = LabelMeta(
        name="future_mfe",
        version="1.0.0",
        description="Maximum Favorable Excursion over horizon",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close", "high", "low"),
        output_columns=("future_mfe",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        _, mfe, _, _ = future_path_stats(
            frame["close"].to_numpy(),
            frame["high"].to_numpy(),
            frame["low"].to_numpy(),
            horizon=h,
        )
        return with_open_time(frame, pl.Series("future_mfe", mfe))


@register_label
class FutureMAE(Label):
    meta = LabelMeta(
        name="future_mae",
        version="1.0.0",
        description="Maximum Adverse Excursion over horizon",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close", "high", "low"),
        output_columns=("future_mae",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        _, _, mae, _ = future_path_stats(
            frame["close"].to_numpy(),
            frame["high"].to_numpy(),
            frame["low"].to_numpy(),
            horizon=h,
        )
        return with_open_time(frame, pl.Series("future_mae", mae))


@register_label
class FutureVWAPDeviation(Label):
    meta = LabelMeta(
        name="future_vwap_deviation",
        version="1.0.0",
        description="Future close vs rolling VWAP deviation",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close", "volume"),
        output_columns=("future_vwap_deviation",),
        parameters={"horizon": _horizon(), "window": 20},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        w = int(self.meta.parameters.get("window", 20))
        pv = (pl.col("close") * pl.col("volume")).rolling_sum(w)
        vv = pl.col("volume").rolling_sum(w)
        vwap = pv / vv
        fut_close = pl.col("close").shift(-h)
        return with_open_time(
            frame, ((fut_close - vwap.shift(-h)) / vwap.shift(-h)).alias("future_vwap_deviation")
        )


@register_label
class FutureSpread(Label):
    meta = LabelMeta(
        name="future_spread",
        version="1.0.0",
        description="Future high-low relative spread",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("high", "low", "close"),
        output_columns=("future_spread",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        spread = (pl.col("high") - pl.col("low")) / pl.col("close")
        return with_open_time(frame, spread.shift(-h).alias("future_spread"))


@register_label
class FutureLiquidity(Label):
    meta = LabelMeta(
        name="future_liquidity",
        version="1.0.0",
        description="Future Amihud-like illiquidity inverse proxy (volume / |ret|)",
        category="regression",
        prediction_horizon=_horizon(),
        required_inputs=("close", "volume"),
        output_columns=("future_liquidity",),
        parameters={"horizon": _horizon()},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", _horizon()))
        ret = pl.col("close").pct_change().abs().clip(1e-9, None)
        liq = pl.col("volume") / ret
        return with_open_time(frame, liq.shift(-h).alias("future_liquidity"))
