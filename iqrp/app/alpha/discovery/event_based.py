"""Event-based discovery templates.

CRITICAL:
- Event windows use only information available at/after the event timestamp
  in a causal way for *signal formation at t*: pre-event features may use
  past data; post-event response measurement for research targets must not
  leak into the signal itself.
- Statistical significance alone ≠ alpha.
- Must track economic_hypothesis.
"""

from __future__ import annotations

import numpy as np

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.discovery.symbolic import as_float1d, lag, rolling_mean, rolling_std


def event_impulse_signal(
    event_mask: np.ndarray,
    *,
    decay: float = 0.5,
    horizon: int = 5,
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    name: str = "event_impulse",
    economic_hypothesis: str | None = None,
) -> AlphaSignal:
    """Causal exponential decay impulse after events.

    At time t, signal aggregates past events only:
    ``sum_k decay^k * event[t-k]`` for k=0..horizon-1.
    No future events enter the value at t.
    """
    ev = np.asarray(event_mask, dtype=np.float64)
    if ev.ndim != 1:
        raise ValueError("event_mask must be 1-D")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not (0 < decay <= 1):
        raise ValueError("decay must be in (0, 1]")
    n = len(ev)
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        s = 0.0
        for k in range(horizon):
            j = t - k
            if j < 0:
                break
            if np.isfinite(ev[j]) and ev[j] != 0:
                s += float(ev[j]) * (decay**k)
        out[t] = s
    hyp = economic_hypothesis or (
        "Discrete corporate or macro events can temporarily shift risk premia and "
        "liquidity; an exponentially decaying impulse encodes lingering post-event "
        "effects as a research candidate, not proven alpha."
    )
    definition = SignalDefinition(
        name=name,
        version="1.0.0",
        formula=f"sum_k decay^k * event[t-k], k<={horizon - 1}",
        features=("event_mask",),
        lookback=horizon,
        horizon=1,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="unknown",
        economic_hypothesis=hyp,
        owner=owner,
        signal_type="event",
        parameters={"decay": decay, "horizon": horizon},
        tags=("event", "impulse", "candidate"),
    )
    return AlphaSignal(
        values=out,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "event_impulse",
            "claims_profitability": False,
        },
    )


def surprise_signal(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    lookback: int = 60,
    owner: str = "research",
    universe: str = "default",
    frequency: str = "1d",
    name: str = "event_surprise",
    economic_hypothesis: str | None = None,
) -> AlphaSignal:
    """Standardized surprise: (actual - expected) / trailing std of surprises.

    Point-in-time: standardization uses past surprises only (lagged).
    """
    a = as_float1d(actual)
    e = as_float1d(expected)
    if len(a) != len(e):
        raise ValueError("actual and expected must have equal length")
    raw = a - e
    hist = lag(raw, 1)
    sd = rolling_std(hist, lookback, min_periods=max(5, lookback // 3))
    out = np.full_like(raw, np.nan)
    mask = np.isfinite(raw) & np.isfinite(sd) & (sd > 1e-12)
    out[mask] = raw[mask] / sd[mask]
    hyp = economic_hypothesis or (
        "Announcement surprises relative to consensus can move prices when "
        "information is not fully anticipated; standardized surprise is a "
        "candidate construct requiring economic validation."
    )
    definition = SignalDefinition(
        name=name,
        version="1.0.0",
        formula=f"(actual-expected)/rolling_std(lag(surprise), {lookback})",
        features=("actual", "expected"),
        lookback=lookback,
        horizon=1,
        universe=universe,
        frequency=frequency,
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=hyp,
        owner=owner,
        signal_type="event",
        parameters={"lookback": lookback},
        tags=("event", "surprise", "candidate"),
    )
    return AlphaSignal(
        values=out,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "surprise",
            "claims_profitability": False,
        },
    )


def earnings_drift_proxy(
    returns: np.ndarray,
    event_mask: np.ndarray,
    *,
    post_window: int = 5,
    owner: str = "research",
) -> AlphaSignal:
    """Delayed PEAD proxy: CAR available only after the post-event window completes.

    At time t, uses events at ``t - post_window`` so the CAR window
    ``[event+1 .. t]`` contains no returns beyond t (no future leakage).
    """
    r = as_float1d(returns)
    ev = np.asarray(event_mask, dtype=np.float64)
    if len(r) != len(ev):
        raise ValueError("returns and event_mask length mismatch")
    if post_window < 1:
        raise ValueError("post_window must be >= 1")
    out = np.full(len(r), np.nan, dtype=np.float64)
    for t in range(len(r)):
        event_t = t - post_window
        if event_t < 0:
            continue
        if not (np.isfinite(ev[event_t]) and ev[event_t] != 0):
            out[t] = 0.0
            continue
        window = r[event_t + 1 : t + 1]
        finite = window[np.isfinite(window)]
        out[t] = float(np.sum(finite)) if finite.size else float("nan")
    definition = SignalDefinition(
        name="event_pead_proxy",
        version="1.0.0",
        formula=f"CAR(event, post_window={post_window}) available only after window completes",
        features=("returns", "event_mask"),
        lookback=post_window,
        horizon=1,
        universe="default",
        frequency="1d",
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=(
            "Post-earnings announcement drift literature suggests delayed incorporation "
            "of earnings news; this delayed CAR proxy is a research candidate only "
            "and does not claim profitability. Statistical significance alone ≠ alpha."
        ),
        owner=owner,
        signal_type="event",
        parameters={"post_window": post_window},
        tags=("event", "pead", "candidate"),
    )
    return AlphaSignal(
        values=out,
        name=definition.name,
        definition_id=definition.definition_id,
        metadata={
            "definition": definition.to_dict(),
            "template": "earnings_drift_proxy",
            "claims_profitability": False,
            "mean_level": float(np.nanmean(rolling_mean(out, 20))) if len(out) else float("nan"),
        },
    )
