"""Reference RESEARCH signals: momentum, mean-reversion, breakout (+ helpers).

These are reference implementations for pipeline validation — NOT profitable strategies.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.signals import SignalRegistry, SignalSpec
from iqrp.app.backtesting.alpha_research.types import SignalKind


def _sign_thresh(x: pd.Series, thr: float = 0.0) -> pd.Series:
    out = pd.Series(0.0, index=x.index, dtype=np.float64)
    out = out.mask(x > thr, 1.0)
    out = out.mask(x < -thr, -1.0)
    out = out.where(x.notna(), np.nan)
    return out


def momentum_signal(
    frame: pd.DataFrame, features: Mapping[str, pd.Series], spec: SignalSpec
) -> pd.Series:
    """+1 if momentum feature > 0 else -1 (0 if flat/nan)."""
    feat = features.get("momentum")
    if feat is None:
        feat = next(iter(features.values()))
    thr = float(spec.parameters.get("threshold", 0.0))
    return _sign_thresh(feat, thr)


def mean_reversion_signal(
    frame: pd.DataFrame, features: Mapping[str, pd.Series], spec: SignalSpec
) -> pd.Series:
    """Fade rolling z-score / RSI extremes."""
    if "rolling_zscore" in features:
        z = features["rolling_zscore"]
        entry = float(spec.parameters.get("entry_z", 1.0))
        out = pd.Series(0.0, index=frame.index, dtype=np.float64)
        out = out.mask(z > entry, -1.0)
        out = out.mask(z < -entry, 1.0)
        return out.where(z.notna(), np.nan)
    rsi = features.get("RSI")
    if rsi is None:
        raise ValueError("mean_reversion requires rolling_zscore or RSI feature")
    hi = float(spec.parameters.get("rsi_high", 70))
    lo = float(spec.parameters.get("rsi_low", 30))
    out = pd.Series(0.0, index=frame.index, dtype=np.float64)
    out = out.mask(rsi > hi, -1.0)
    out = out.mask(rsi < lo, 1.0)
    return out.where(rsi.notna(), np.nan)


def breakout_signal(
    frame: pd.DataFrame, features: Mapping[str, pd.Series], spec: SignalSpec
) -> pd.Series:
    """Breakout vs trailing high/low (causal: compare close to prior window max/min)."""
    lb = max(int(spec.parameters.get("lookback", 20)), 2)
    c = pd.Series(frame["close"], dtype=np.float64)
    # prior window excludes current bar → causal
    prior_high = c.shift(1).rolling(lb, min_periods=lb).max()
    prior_low = c.shift(1).rolling(lb, min_periods=lb).min()
    out = pd.Series(0.0, index=frame.index, dtype=np.float64)
    out = out.mask(c > prior_high, 1.0)
    out = out.mask(c < prior_low, -1.0)
    return out


def volatility_signal(
    frame: pd.DataFrame, features: Mapping[str, pd.Series], spec: SignalSpec
) -> pd.Series:
    """Risk-off when vol high vs its rolling median (research diagnostic)."""
    vol = features.get("volatility")
    if vol is None:
        raise ValueError("volatility signal needs volatility feature")
    lb = max(int(spec.parameters.get("vol_lookback", 20)), 2)
    med = vol.rolling(lb, min_periods=lb).median()
    out = pd.Series(0.0, index=frame.index, dtype=np.float64)
    out = out.mask(vol > med, -1.0)
    out = out.mask(vol < med, 1.0)
    return out.where(vol.notna(), np.nan)


def volume_signal(
    frame: pd.DataFrame, features: Mapping[str, pd.Series], spec: SignalSpec
) -> pd.Series:
    """Align with price when volume expands (research)."""
    vc = features.get("volume_change")
    mom = features.get("momentum")
    if vc is None or mom is None:
        raise ValueError("volume signal needs volume_change and momentum")
    thr = float(spec.parameters.get("volume_threshold", 0.0))
    out = _sign_thresh(mom, 0.0)
    out = out.where(vc > thr, 0.0)
    return out


def trend_signal(
    frame: pd.DataFrame, features: Mapping[str, pd.Series], spec: SignalSpec
) -> pd.Series:
    """Price vs MA: above → long, below → short."""
    ma = features.get("moving_average")
    if ma is None:
        ma = features.get("EMA")
    if ma is None:
        raise ValueError("trend signal needs moving_average or EMA")
    c = pd.Series(frame["close"], dtype=np.float64)
    return _sign_thresh(c - ma, 0.0)


def price_action_signal(
    frame: pd.DataFrame, features: Mapping[str, pd.Series], spec: SignalSpec
) -> pd.Series:
    """Wide range continuation (research)."""
    rng = features.get("range")
    mom = features.get("returns")
    if mom is None:
        mom = features.get("momentum")
    if rng is None or mom is None:
        raise ValueError("price_action needs range and returns/momentum")
    thr = float(spec.parameters.get("range_threshold", rng.median(skipna=True) or 0.0))
    out = _sign_thresh(mom, 0.0)
    out = out.where(rng >= thr, 0.0)
    return out


def register_reference_signals(reg: SignalRegistry) -> None:
    items = [
        (
            SignalSpec(
                "momentum_signal",
                description="Sign of N-bar momentum",
                feature_ids=("momentum",),
                kind=SignalKind.CATEGORICAL,
                family="momentum",
                parameters={"lookback": 20, "threshold": 0.0},
            ),
            momentum_signal,
        ),
        (
            SignalSpec(
                "mean_reversion_signal",
                description="Fade rolling z-score extremes",
                feature_ids=("rolling_zscore",),
                kind=SignalKind.CATEGORICAL,
                family="mean_reversion",
                parameters={"lookback": 20, "entry_z": 1.0},
            ),
            mean_reversion_signal,
        ),
        (
            SignalSpec(
                "breakout_signal",
                description="Break prior N-bar high/low",
                feature_ids=("momentum",),  # unused; uses price path causally
                kind=SignalKind.CATEGORICAL,
                family="breakout",
                parameters={"lookback": 20},
            ),
            breakout_signal,
        ),
        (
            SignalSpec(
                "volatility_signal",
                description="Vol regime tilt",
                feature_ids=("volatility",),
                family="volatility",
                parameters={"lookback": 20},
            ),
            volatility_signal,
        ),
        (
            SignalSpec(
                "volume_signal",
                description="Volume-confirmed momentum",
                feature_ids=("volume_change", "momentum"),
                family="volume",
                parameters={"lookback": 10},
            ),
            volume_signal,
        ),
        (
            SignalSpec(
                "trend_signal",
                description="Price vs moving average",
                feature_ids=("moving_average",),
                family="trend",
                parameters={"lookback": 20},
            ),
            trend_signal,
        ),
        (
            SignalSpec(
                "price_action_signal",
                description="Range-confirmed return sign",
                feature_ids=("range", "returns"),
                family="price_action",
                parameters={"lookback": 1},
            ),
            price_action_signal,
        ),
    ]
    for spec, fn in items:
        reg.register(spec, fn, overwrite=True)


__all__ = ["register_reference_signals"]
