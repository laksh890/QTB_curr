"""Future-target construction for predictive research (no trading signals)."""

from __future__ import annotations

import numpy as np
import polars as pl

from iqrp.app.features.research.config import ResearchSettings


def select_feature_columns(frame: pl.DataFrame, settings: ResearchSettings) -> list[str]:
    ts = settings.columns.timestamp
    close = settings.columns.close
    exclude = {ts, close, "open", "high", "low", "volume", "symbol", "exchange", "timeframe"}
    cols = [
        c
        for c, dt in zip(frame.columns, frame.dtypes, strict=False)
        if c not in exclude and dt.is_numeric()
    ]
    prefix = settings.columns.feature_prefix
    if prefix:
        cols = [c for c in cols if c.startswith(prefix)]
    return cols


def build_targets(frame: pl.DataFrame, settings: ResearchSettings) -> pl.DataFrame:
    """Construct supervised research targets from price columns."""
    close = settings.columns.close
    ts = settings.columns.timestamp
    if close not in frame.columns:
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError(
            f"Missing close column '{close}' for target construction",
            code="RESEARCH_MISSING_CLOSE",
        )
    h = settings.targets.return_horizon
    vw = settings.targets.volatility_window
    dw = settings.targets.drawdown_window
    thr = settings.targets.direction_threshold
    rw = settings.targets.regime_vol_window
    q_lo, q_hi = settings.targets.regime_quantiles[0], settings.targets.regime_quantiles[1]

    fut_ret = pl.col(close).shift(-h) / pl.col(close) - 1.0
    # Future realized vol over next ``vw`` bars of returns
    ret = pl.col(close).pct_change()
    fut_vol = ret.shift(-vw).rolling_std(vw)
    # Future drawdown: min of forward cumulative returns path proxy
    # Use rolling min of future returns window as drawdown intensity proxy.
    fut_dd = fut_ret.rolling_min(dw).shift(-dw)
    fut_dir = (fut_ret > thr).cast(pl.Float64)
    # Regime from trailing vol terciles, shifted so label is future regime
    trail_vol = ret.rolling_std(rw)
    # Assign regime using quantiles computed in Python for stability
    vol_np = frame.select(trail_vol.alias("v")).to_series().to_numpy()
    finite = vol_np[np.isfinite(vol_np)]
    if len(finite) > 10:
        lo = float(np.quantile(finite, q_lo))
        hi = float(np.quantile(finite, q_hi))
    else:
        lo, hi = 0.0, 1.0
    regime = (
        pl.when(trail_vol.is_null())
        .then(None)
        .when(trail_vol <= lo)
        .then(0.0)
        .when(trail_vol >= hi)
        .then(2.0)
        .otherwise(1.0)
        .shift(-h)
    )

    out = frame.select(
        ([pl.col(ts)] if ts in frame.columns else [])
        + [
            fut_ret.alias("future_return"),
            fut_vol.alias("future_volatility"),
            fut_dd.alias("future_drawdown"),
            fut_dir.alias("future_direction"),
            regime.alias("future_regime"),
        ]
    )
    return out


TARGET_NAMES = (
    "future_return",
    "future_volatility",
    "future_drawdown",
    "future_direction",
    "future_regime",
)
