"""Time-series discovery templates: momentum, mean-reversion, trend, vol, volume.

CRITICAL:
- Templates produce SignalDefinition + signal arrays from price/return inputs
  without claiming profitability.
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Point-in-time: only past windows are used in signal computation.
- Must track economic_hypothesis on every SignalDefinition.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.discovery.symbolic import (
    as_float1d,
    lag,
    rolling_mean,
    rolling_std,
    rolling_sum,
    zscore,
)


def _returns_from_prices(prices: np.ndarray) -> np.ndarray:
    p = as_float1d(prices)
    out = np.full_like(p, np.nan)
    # Point-in-time simple return: p[t]/p[t-1] - 1 (uses past price only)
    prev = lag(p, 1)
    mask = np.isfinite(p) & np.isfinite(prev) & (np.abs(prev) > 1e-12)
    out[mask] = p[mask] / prev[mask] - 1.0
    return out


def _ensure_returns(
    data: np.ndarray,
    *,
    input_kind: str = "returns",
) -> np.ndarray:
    if input_kind == "returns":
        return as_float1d(data)
    if input_kind == "prices":
        return _returns_from_prices(data)
    raise ValueError(f"input_kind must be 'returns' or 'prices', got {input_kind!r}")


def momentum_signal(
    data: np.ndarray,
    lookback: int = 20,
    *,
    input_kind: str = "returns",
    timestamps: np.ndarray | None = None,
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    horizon: int = 1,
) -> AlphaSignal:
    """Trailing momentum = sum of past ``lookback`` returns (inclusive of t).

    Research candidate only — does not claim profitability.
    Point-in-time: uses ``returns[t-lookback+1 : t+1]`` only.
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    r = _ensure_returns(data, input_kind=input_kind)
    values = rolling_sum(r, lookback, min_periods=lookback)
    definition = SignalDefinition(
        name="ts_momentum",
        version="1.0.0",
        formula=f"sum(returns, {lookback})",
        features=("returns",),
        lookback=lookback,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=(
            "Short-horizon return continuation can arise from underreaction to "
            "information and gradual capital flows; this template measures trailing "
            "cumulative returns as a research candidate, not proven alpha."
        ),
        owner=owner,
        signal_type="momentum",
        parameters={"lookback": lookback, "input_kind": input_kind},
        tags=("time_series", "momentum", "candidate"),
    )
    return AlphaSignal(
        values=values,
        timestamps=timestamps,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "momentum",
            "claims_profitability": False,
        },
    )


def mean_reversion_signal(
    data: np.ndarray,
    lookback: int = 20,
    *,
    input_kind: str = "returns",
    timestamps: np.ndarray | None = None,
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    horizon: int = 1,
) -> AlphaSignal:
    """Negative trailing z-score of returns (fade recent extremes).

    Research candidate only — does not claim profitability.
    Point-in-time z-score over past window.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2 for mean-reversion z-score")
    r = _ensure_returns(data, input_kind=input_kind)
    values = -zscore(r, lookback, min_periods=lookback)
    definition = SignalDefinition(
        name="ts_mean_reversion",
        version="1.0.0",
        formula=f"-zscore(returns, {lookback})",
        features=("returns",),
        lookback=lookback,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=(
            "Transitory liquidity shocks and inventory effects can push prices away "
            "from short-term fair value, motivating fade-of-extreme return z-scores "
            "as a research candidate. Not approved alpha by construction."
        ),
        owner=owner,
        signal_type="mean_reversion",
        parameters={"lookback": lookback, "input_kind": input_kind},
        tags=("time_series", "mean_reversion", "candidate"),
    )
    return AlphaSignal(
        values=values,
        timestamps=timestamps,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "mean_reversion",
            "claims_profitability": False,
        },
    )


def trend_signal(
    data: np.ndarray,
    lookback_fast: int = 10,
    lookback_slow: int = 40,
    *,
    input_kind: str = "prices",
    timestamps: np.ndarray | None = None,
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    horizon: int = 1,
) -> AlphaSignal:
    """Fast vs slow trailing mean of prices (or of cumulative returns proxy).

    Point-in-time: both means use only past windows.
    """
    if lookback_fast < 1 or lookback_slow < 1:
        raise ValueError("lookbacks must be >= 1")
    if lookback_fast >= lookback_slow:
        raise ValueError("lookback_fast must be < lookback_slow")
    if input_kind == "prices":
        series = as_float1d(data)
        feat = "prices"
    else:
        # Use cumulative sum of returns as a price proxy (PIT causal)
        r = as_float1d(data)
        series = np.cumsum(np.nan_to_num(r, nan=0.0))
        feat = "returns"
    fast = rolling_mean(series, lookback_fast, min_periods=lookback_fast)
    slow = rolling_mean(series, lookback_slow, min_periods=lookback_slow)
    values = fast - slow
    definition = SignalDefinition(
        name="ts_trend",
        version="1.0.0",
        formula=f"sma({lookback_fast}) - sma({lookback_slow})",
        features=(feat,),
        lookback=lookback_slow,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=(
            "Persistent risk premia and trend-following capital can create medium-horizon "
            "drift in prices; dual moving-average spread is a research template only."
        ),
        owner=owner,
        signal_type="trend",
        parameters={
            "lookback_fast": lookback_fast,
            "lookback_slow": lookback_slow,
            "input_kind": input_kind,
        },
        tags=("time_series", "trend", "candidate"),
    )
    return AlphaSignal(
        values=values,
        timestamps=timestamps,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "trend",
            "claims_profitability": False,
        },
    )


def volatility_signal(
    data: np.ndarray,
    lookback: int = 20,
    *,
    input_kind: str = "returns",
    invert: bool = True,
    timestamps: np.ndarray | None = None,
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    horizon: int = 1,
) -> AlphaSignal:
    """Trailing realized volatility; optionally inverted (low-vol preference).

    Research candidate only — no profitability claim.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    r = _ensure_returns(data, input_kind=input_kind)
    vol = rolling_std(r, lookback, min_periods=lookback)
    values = -vol if invert else vol
    definition = SignalDefinition(
        name="ts_volatility",
        version="1.0.0",
        formula=f"{'-' if invert else ''}rolling_std(returns, {lookback})",
        features=("returns",),
        lookback=lookback,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="positive" if invert else "negative",
        economic_hypothesis=(
            "Risk-based pricing and leverage constraints can link realized volatility "
            "to subsequent returns; this measures trailing vol as a candidate feature, "
            "not approved alpha. Statistical significance alone ≠ alpha."
        ),
        owner=owner,
        signal_type="volatility",
        parameters={"lookback": lookback, "invert": invert, "input_kind": input_kind},
        tags=("time_series", "volatility", "candidate"),
    )
    return AlphaSignal(
        values=values,
        timestamps=timestamps,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "volatility",
            "claims_profitability": False,
        },
    )


def volume_signal(
    volume: np.ndarray,
    lookback: int = 20,
    *,
    timestamps: np.ndarray | None = None,
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    horizon: int = 1,
) -> AlphaSignal:
    """Trailing volume z-score (abnormal activity) as a research candidate."""
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    v = as_float1d(volume)
    values = zscore(v, lookback, min_periods=lookback)
    definition = SignalDefinition(
        name="ts_volume",
        version="1.0.0",
        formula=f"zscore(volume, {lookback})",
        features=("volume",),
        lookback=lookback,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="unknown",
        economic_hypothesis=(
            "Abnormal trading activity can coincide with information arrival or "
            "liquidity demand; volume z-score is a discovery template without a "
            "profitability claim. Historical Sharpe alone cannot approve."
        ),
        owner=owner,
        signal_type="volume",
        parameters={"lookback": lookback},
        tags=("time_series", "volume", "candidate"),
    )
    return AlphaSignal(
        values=values,
        timestamps=timestamps,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "volume",
            "claims_profitability": False,
        },
    )


def build_time_series_candidates(
    returns: np.ndarray,
    *,
    volume: np.ndarray | None = None,
    prices: np.ndarray | None = None,
    momentum_lookbacks: tuple[int, ...] = (10, 20, 60),
    mean_rev_lookbacks: tuple[int, ...] = (5, 10, 20),
    **kwargs: Any,
) -> list[AlphaSignal]:
    """Orchestrate standard TS templates into a candidate list (not alpha)."""
    out: list[AlphaSignal] = []
    for lb in momentum_lookbacks:
        out.append(momentum_signal(returns, lookback=lb, **kwargs))
    for lb in mean_rev_lookbacks:
        out.append(mean_reversion_signal(returns, lookback=lb, **kwargs))
    if prices is not None:
        out.append(trend_signal(prices, input_kind="prices", **kwargs))
    else:
        out.append(trend_signal(returns, input_kind="returns", **kwargs))
    out.append(volatility_signal(returns, **kwargs))
    if volume is not None:
        out.append(volume_signal(volume, **kwargs))
    return out
