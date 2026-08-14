"""Dataset frequency availability for horizon research."""

from __future__ import annotations

from typing import Any

import pandas as pd

from iqrp.app.backtesting.data.schema import infer_frequency
from iqrp.app.backtesting.horizon.parse import (
    availability_reason,
    can_derive,
    parse_timeframe,
)
from iqrp.app.backtesting.horizon.types import HorizonStatus, Timeframe


def detect_native_frequency(frame: pd.DataFrame) -> Timeframe:
    """Infer native bar frequency from a normalized OHLCV frame."""
    if frame is None or len(frame) == 0:
        raise ValueError("empty frame")
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    if "instrument" in frame.columns:
        # use the most common instrument's spacing
        counts = frame["instrument"].value_counts()
        inst = str(counts.index[0])
        ts = pd.to_datetime(frame.loc[frame["instrument"] == inst, "timestamp"], utc=True)
    label = infer_frequency(ts)
    # infer_frequency may return strings like 1d / 1h / irregular
    if not label or str(label).lower() in {"irregular", "unknown", "none"}:
        # fallback: median delta
        deltas = ts.sort_values().diff().dropna().dt.total_seconds()
        if deltas.empty:
            raise ValueError("cannot infer frequency")
        sec = float(deltas.median())
        return parse_timeframe(sec)
    try:
        return parse_timeframe(str(label))
    except ValueError:
        deltas = ts.sort_values().diff().dropna().dt.total_seconds()
        sec = float(deltas.median()) if not deltas.empty else 86400.0
        return parse_timeframe(sec)


def check_horizon_availability(
    native: Timeframe | str,
    requested_data_tf: Timeframe | str,
) -> dict[str, Any]:
    """Return availability gate for a requested data timeframe."""
    n = parse_timeframe(native)
    r = parse_timeframe(requested_data_tf)
    ok = can_derive(n, r)
    return {
        "available": bool(ok),
        "native": str(n),
        "requested": str(r),
        "status": HorizonStatus.UNAVAILABLE.value if not ok else "AVAILABLE",
        "reason": availability_reason(n, r),
    }


def filter_available_timeframes(
    native: Timeframe | str,
    candidates: list[str],
) -> dict[str, Any]:
    """Split candidates into available vs unavailable."""
    available: list[str] = []
    unavailable: list[dict[str, Any]] = []
    for c in candidates:
        gate = check_horizon_availability(native, c)
        if gate["available"]:
            available.append(str(parse_timeframe(c)))
        else:
            unavailable.append(gate)
    return {
        "native": str(parse_timeframe(native)),
        "available": available,
        "unavailable": unavailable,
    }


__all__ = [
    "check_horizon_availability",
    "detect_native_frequency",
    "filter_available_timeframes",
]
