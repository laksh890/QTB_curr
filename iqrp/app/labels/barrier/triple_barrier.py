"""Full Triple Barrier Method label generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from iqrp.app.labels._utils import atr, with_open_time
from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.registry import register_label
from iqrp.app.labels.config import LabelSettings, TripleBarrierConfig

HitType = Literal["upper", "lower", "time"]


@dataclass(frozen=True, slots=True)
class TripleBarrierResult:
    hit_type: np.ndarray
    hit_time: np.ndarray
    label_return: np.ndarray
    upper_barrier: np.ndarray
    lower_barrier: np.ndarray


def compute_triple_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    upper: np.ndarray,
    lower: np.ndarray,
    horizon: int,
) -> TripleBarrierResult:
    """Path-dependent triple barrier labeling.

    Barriers are absolute prices. ``upper``/``lower`` are per-row at entry time.
    Hit codes: 1=upper, -1=lower, 0=time.
    """
    n = len(close)
    hit_type = np.full(n, np.nan)
    hit_time = np.full(n, np.nan)
    rets = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(close[i]) or not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
            continue
        end = min(n - 1, i + horizon)
        if end <= i:
            continue
        hit = 0  # time
        t_hit = end - i
        exit_px = close[end]
        for j in range(i + 1, end + 1):
            if high[j] >= upper[i]:
                hit = 1
                t_hit = j - i
                exit_px = upper[i]
                break
            if low[j] <= lower[i]:
                hit = -1
                t_hit = j - i
                exit_px = lower[i]
                break
        hit_type[i] = float(hit)
        hit_time[i] = float(t_hit)
        rets[i] = exit_px / close[i] - 1.0
    return TripleBarrierResult(hit_type, hit_time, rets, upper, lower)


def build_barriers(
    frame: pl.DataFrame,
    cfg: TripleBarrierConfig,
) -> tuple[np.ndarray, np.ndarray]:
    close = frame["close"].to_numpy().astype(np.float64)
    if cfg.barrier_mode == "fixed":
        upper = close * (1.0 + cfg.fixed_upper * cfg.upper_mult)
        lower = close * (1.0 - cfg.fixed_lower * cfg.lower_mult)
        return upper, lower
    if cfg.barrier_mode == "volatility":
        vol = (
            frame.select(pl.col("close").pct_change().rolling_std(cfg.vol_window).alias("v"))
            .to_series()
            .to_numpy()
            .astype(np.float64)
        )
        width = cfg.vol_multiplier * vol
        upper = close * (1.0 + width * cfg.upper_mult)
        lower = close * (1.0 - width * cfg.lower_mult)
        return upper, lower
    # atr (default)
    atr_s = atr(frame, cfg.atr_window).to_numpy().astype(np.float64)
    width = cfg.atr_multiplier * atr_s
    upper = close + width * cfg.upper_mult
    lower = close - width * cfg.lower_mult
    return upper, lower


@register_label
class TripleBarrierLabel(Label):
    meta = LabelMeta(
        name="triple_barrier",
        version="1.0.0",
        description="Triple barrier labels: hit type, hit time, barrier return",
        category="barrier",
        prediction_horizon=LabelSettings.default().triple_barrier.horizon,
        required_inputs=("open", "high", "low", "close"),
        output_columns=(
            "tb_hit_type",
            "tb_hit_time",
            "tb_return",
            "tb_upper",
            "tb_lower",
        ),
        parameters=LabelSettings.default().triple_barrier.model_dump(),
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        cfg = TripleBarrierConfig.model_validate(self.meta.parameters)
        upper, lower = build_barriers(frame, cfg)
        result = compute_triple_barrier(
            frame["close"].to_numpy().astype(np.float64),
            frame["high"].to_numpy().astype(np.float64),
            frame["low"].to_numpy().astype(np.float64),
            upper=upper,
            lower=lower,
            horizon=cfg.horizon,
        )
        return with_open_time(
            frame,
            pl.Series("tb_hit_type", result.hit_type),
            pl.Series("tb_hit_time", result.hit_time),
            pl.Series("tb_return", result.label_return),
            pl.Series("tb_upper", result.upper_barrier),
            pl.Series("tb_lower", result.lower_barrier),
        )


def triple_barrier_frame(
    frame: pl.DataFrame,
    settings: LabelSettings | None = None,
    **overrides: Any,
) -> pl.DataFrame:
    """Functional API for triple barrier with optional parameter overrides."""
    settings = settings or LabelSettings.default()
    params = {**settings.triple_barrier.model_dump(), **overrides}
    label = TripleBarrierLabel()
    label.meta = LabelMeta(
        name="triple_barrier",
        version="1.0.0",
        description=label.meta.description,
        category="barrier",
        prediction_horizon=int(params["horizon"]),
        required_inputs=label.meta.required_inputs,
        output_columns=label.meta.output_columns,
        parameters=params,
    )
    return label.compute(frame)
