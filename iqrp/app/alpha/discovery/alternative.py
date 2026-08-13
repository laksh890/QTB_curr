"""Alternative-data discovery templates.

CRITICAL:
- Alternative series must be aligned point-in-time (publication lag applied).
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Must track economic_hypothesis on SignalDefinition.
"""

from __future__ import annotations

import numpy as np

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.discovery.symbolic import as_float1d, lag, rolling_mean, zscore


def apply_publication_lag(series: np.ndarray, lag_periods: int) -> np.ndarray:
    """Shift alternative data by publication lag so only past-known values appear at t."""
    if lag_periods < 0:
        raise ValueError("publication lag must be >= 0 (negative would leak future)")
    return lag(as_float1d(series), lag_periods)


def alternative_zscore_signal(
    alt_series: np.ndarray,
    *,
    lookback: int = 60,
    publication_lag: int = 1,
    name: str = "alt_zscore",
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    horizon: int = 1,
    economic_hypothesis: str | None = None,
) -> AlphaSignal:
    """Z-score of lag-adjusted alternative series (PIT)."""
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    delayed = apply_publication_lag(alt_series, publication_lag)
    values = zscore(delayed, lookback, min_periods=lookback)
    hyp = economic_hypothesis or (
        "Alternative datasets (satellite, web, credit-card, etc.) can proxy real-activity "
        "or sentiment before it is fully reflected in prices, subject to publication lag; "
        "this is a research candidate, not approved alpha."
    )
    definition = SignalDefinition(
        name=name,
        version="1.0.0",
        formula=f"zscore(lag(alt, {publication_lag}), {lookback})",
        features=("alt_series",),
        lookback=lookback + publication_lag,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="unknown",
        economic_hypothesis=hyp,
        owner=owner,
        signal_type="alternative",
        parameters={"lookback": lookback, "publication_lag": publication_lag},
        tags=("alternative", "zscore", "candidate"),
    )
    return AlphaSignal(
        values=values,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "alternative_zscore",
            "claims_profitability": False,
            "publication_lag": publication_lag,
        },
    )


def alternative_change_signal(
    alt_series: np.ndarray,
    *,
    change_window: int = 5,
    publication_lag: int = 1,
    name: str = "alt_change",
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    horizon: int = 1,
    economic_hypothesis: str | None = None,
) -> AlphaSignal:
    """Change in lag-adjusted alternative series vs trailing mean."""
    delayed = apply_publication_lag(alt_series, publication_lag)
    mu = rolling_mean(delayed, change_window, min_periods=change_window)
    values = delayed - mu
    hyp = economic_hypothesis or (
        "Innovations in alternative activity measures may lead traditional macro or "
        "fundamental prints; change-vs-trailing-mean is a candidate template only."
    )
    definition = SignalDefinition(
        name=name,
        version="1.0.0",
        formula=f"lag(alt,{publication_lag}) - sma(lag(alt), {change_window})",
        features=("alt_series",),
        lookback=change_window + publication_lag,
        horizon=horizon,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=hyp,
        owner=owner,
        signal_type="alternative",
        parameters={"change_window": change_window, "publication_lag": publication_lag},
        tags=("alternative", "change", "candidate"),
    )
    return AlphaSignal(
        values=values,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "alternative_change",
            "claims_profitability": False,
        },
    )


def sentiment_pressure_signal(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    publication_lag: int = 0,
    lookback: int = 10,
    name: str = "alt_sentiment",
    owner: str = "research",
    economic_hypothesis: str | None = None,
) -> AlphaSignal:
    """Net sentiment (pos - neg) smoothed with trailing mean after publication lag."""
    pos = apply_publication_lag(positive, publication_lag)
    neg = apply_publication_lag(negative, publication_lag)
    raw = pos - neg
    values = rolling_mean(raw, lookback, min_periods=max(1, lookback // 2))
    hyp = economic_hypothesis or (
        "Aggregated textual or social sentiment can capture shifts in risk appetite "
        "and attention; net sentiment pressure is exploratory and not alpha by itself. "
        "Historical Sharpe alone cannot approve."
    )
    definition = SignalDefinition(
        name=name,
        version="1.0.0",
        formula=f"sma(lag(pos-neg, {publication_lag}), {lookback})",
        features=("positive", "negative"),
        lookback=lookback + publication_lag,
        horizon=1,
        universe="default",
        frequency="1d",
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=hyp,
        owner=owner,
        signal_type="alternative",
        parameters={"lookback": lookback, "publication_lag": publication_lag},
        tags=("alternative", "sentiment", "candidate"),
    )
    return AlphaSignal(
        values=values,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "sentiment_pressure",
            "claims_profitability": False,
        },
    )
