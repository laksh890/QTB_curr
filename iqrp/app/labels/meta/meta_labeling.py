"""Meta-labeling framework (primary signal + secondary confirmation)."""

from __future__ import annotations

import numpy as np
import polars as pl

from iqrp.app.labels._utils import with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import register_label
from iqrp.app.labels.config import LabelSettings


@register_label
class MetaLabel(Label):
    """Meta-label: whether a primary directional signal was profitable.

    Requires a ``primary_signal`` column in {-1, 0, 1} when available.
    If missing, a synthetic side from next-bar return sign is used so the
    label pipeline remains runnable on pure OHLCV research frames.
    """

    meta = LabelMeta(
        name="meta_label",
        version="1.0.0",
        description="Secondary meta-label confirming primary signal profitability",
        category="meta",
        prediction_horizon=LabelSettings.default().defaults.horizon,
        required_inputs=("close",),
        output_columns=("meta_label", "meta_return"),
        parameters={
            "horizon": LabelSettings.default().defaults.horizon,
            "primary_signal_column": LabelSettings.default().meta_labeling.primary_signal_column,
            "confirmation_column": LabelSettings.default().meta_labeling.confirmation_column,
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = LabelSettings.default().meta_labeling
        h = int(self.meta.parameters.get("horizon", LabelSettings.default().defaults.horizon))
        sig_col = str(self.meta.parameters.get("primary_signal_column", cfg.primary_signal_column))
        if sig_col in frame.columns:
            side = frame[sig_col].cast(pl.Float64).fill_null(0.0).to_numpy()
        else:
            side = (
                (frame["close"].shift(-1) / frame["close"] - 1.0).sign().fill_null(0.0).to_numpy()
            )
        fut = (frame["close"].shift(-h) / frame["close"] - 1.0).to_numpy()
        meta_ret = side * fut
        meta = np.where(side == 0, np.nan, (meta_ret > 0).astype(float))
        conf_col = self.meta.parameters.get("confirmation_column", cfg.confirmation_column)
        if conf_col and str(conf_col) in frame.columns:
            conf = frame[str(conf_col)].to_numpy()
            meta = np.where(np.isfinite(conf) & (conf <= 0), 0.0, meta)
        return with_open_time(
            frame,
            pl.Series("meta_label", meta),
            pl.Series("meta_return", meta_ret),
        )


@register_label
class ProbabilityLabel(Label):
    meta = LabelMeta(
        name="probability_label",
        version="1.0.0",
        description="Soft probability-style label from sigmoid of scaled future return",
        category="meta",
        prediction_horizon=LabelSettings.default().defaults.horizon,
        required_inputs=("close",),
        output_columns=("probability_label",),
        parameters={"horizon": LabelSettings.default().defaults.horizon, "scale": 50.0},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", 12))
        scale = float(self.meta.parameters.get("scale", 50.0))
        fut = pl.col("close").shift(-h) / pl.col("close") - 1.0
        prob = 1.0 / (1.0 + (-scale * fut).exp())
        return with_open_time(frame, prob.alias("probability_label"))


@register_label
class TradeFilterLabel(Label):
    meta = LabelMeta(
        name="trade_filter_label",
        version="1.0.0",
        description="1 if |future return| exceeds threshold (trade worth taking)",
        category="meta",
        prediction_horizon=LabelSettings.default().defaults.horizon,
        required_inputs=("close",),
        output_columns=("trade_filter_label",),
        parameters={
            "horizon": LabelSettings.default().defaults.horizon,
            "threshold": 0.005,
        },
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        h = int(self.meta.parameters.get("horizon", 12))
        thr = float(self.meta.parameters.get("threshold", 0.005))
        fut = (pl.col("close").shift(-h) / pl.col("close") - 1.0).abs()
        return with_open_time(frame, (fut >= thr).cast(pl.Float64).alias("trade_filter_label"))


def meta_label_frame(
    frame: pl.DataFrame,
    *,
    primary_signal_column: str = "primary_signal",
    horizon: int | None = None,
    confirmation_column: str | None = None,
) -> pl.DataFrame:
    """Functional meta-label API."""
    settings = LabelSettings.default()
    label = MetaLabel()
    label.meta = LabelMeta(
        name="meta_label",
        version="1.0.0",
        description=label.meta.description,
        category="meta",
        prediction_horizon=horizon or settings.defaults.horizon,
        required_inputs=("close",),
        output_columns=("meta_label", "meta_return"),
        parameters={
            "horizon": horizon or settings.defaults.horizon,
            "primary_signal_column": primary_signal_column,
            "confirmation_column": confirmation_column,
        },
    )
    return label.compute(frame)


def secondary_confirmation(
    frame: pl.DataFrame,
    *,
    primary_signal_column: str,
    confirmation_column: str,
) -> pl.DataFrame:
    """Intersect primary signal with confirmation flag."""
    out = frame.with_columns(
        ((pl.col(primary_signal_column) != 0) & (pl.col(confirmation_column) > 0))
        .cast(pl.Float64)
        .alias("secondary_confirmation")
    )
    if "open_time" in out.columns:
        return out.select("open_time", "secondary_confirmation")
    return out.select("secondary_confirmation")
