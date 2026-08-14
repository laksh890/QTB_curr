"""Parse timeframe / holding-period tokens."""

from __future__ import annotations

import re
from typing import Any

from iqrp.app.backtesting.horizon.types import HoldingPeriod, Timeframe

_TF_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")

_UNIT_SECONDS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "1d": 86400.0,
}


def parse_timeframe(token: str | Timeframe | float | int) -> Timeframe:
    """Parse ``1m``, ``5m``, ``15m``, ``30m``, ``1h``, ``4h``, ``1D``, ``daily``."""
    if isinstance(token, Timeframe):
        return token
    if isinstance(token, (int, float)):
        sec = float(token)
        return Timeframe(label=_label_from_seconds(sec), seconds=sec)

    raw = str(token).strip()
    low = raw.lower()
    aliases = {
        "daily": "1D",
        "day": "1D",
        "1day": "1D",
        "d1": "1D",
        "hourly": "1h",
        "minute": "1m",
        "min": "1m",
    }
    raw = aliases.get(low, raw)
    # Preserve D capitalization for day tokens
    m = _TF_RE.match(raw)
    if not m:
        raise ValueError(f"unrecognized timeframe token: {token!r}")
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit in {"d", "day", "days"}:
        label = f"{n}D"
        seconds = n * 86400.0
    else:
        if unit not in _UNIT_SECONDS:
            raise ValueError(f"unrecognized timeframe unit in {token!r}")
        seconds = n * _UNIT_SECONDS[unit]
        label = f"{n}{unit[0]}" if unit[0] != "d" else f"{n}D"
        # normalize common forms
        if unit.startswith("min") or unit == "m":
            label = f"{n}m"
        elif unit.startswith("h"):
            label = f"{n}h"
        elif unit.startswith("s"):
            label = f"{n}s"
    return Timeframe(label=label, seconds=float(seconds))


def _label_from_seconds(seconds: float) -> str:
    s = float(seconds)
    if abs(s - 86400.0) < 1e-6 or s % 86400 == 0 and s >= 86400:
        return f"{int(round(s / 86400.0))}D"
    if s % 3600 == 0 and s >= 3600:
        return f"{int(round(s / 3600.0))}h"
    if s % 60 == 0 and s >= 60:
        return f"{int(round(s / 60.0))}m"
    return f"{int(round(s))}s"


def parse_holding(
    token: str | int | HoldingPeriod | None,
    *,
    bar_seconds: float | None = None,
) -> HoldingPeriod:
    """Parse ``5``, ``5bar``, ``5bars``, ``30m`` holding tokens."""
    if token is None:
        return HoldingPeriod(bars=1, label="1bar")
    if isinstance(token, HoldingPeriod):
        return token
    if isinstance(token, int):
        return HoldingPeriod(bars=int(token), label=f"{int(token)}bar")
    raw = str(token).strip().lower()
    if raw.endswith("bars"):
        n = int(raw[:-4] or "1")
        return HoldingPeriod(bars=n, label=f"{n}bar")
    if raw.endswith("bar"):
        n = int(raw[:-3] or "1")
        return HoldingPeriod(bars=n, label=f"{n}bar")
    if raw.isdigit():
        n = int(raw)
        return HoldingPeriod(bars=n, label=f"{n}bar")
    # time-based holding
    tf = parse_timeframe(raw)
    bars = None
    if bar_seconds and bar_seconds > 0:
        bars = max(1, int(round(tf.seconds / float(bar_seconds))))
    return HoldingPeriod(bars=bars, seconds=tf.seconds, label=str(tf))


def timeframe_rank(tf: Timeframe | str) -> float:
    """Numeric rank for comparisons (seconds)."""
    return float(parse_timeframe(tf).seconds)


def can_derive(native: Timeframe | str, requested: Timeframe | str) -> bool:
    """True if ``requested`` can be obtained from ``native`` without fabricating finer bars.

    Equal or coarser requested timeframes are allowed (resample down).
    Finer than native → False.
    """
    n = parse_timeframe(native).seconds
    r = parse_timeframe(requested).seconds
    # allow small float noise
    return r + 1e-9 >= n


def availability_reason(native: Timeframe | str, requested: Timeframe | str) -> str:
    n = parse_timeframe(native)
    r = parse_timeframe(requested)
    if can_derive(n, r):
        if abs(n.seconds - r.seconds) < 1e-9:
            return f"native frequency {n} matches requested {r}"
        return f"requested {r} is coarser than native {n}; downsampling allowed"
    return (
        f"requested {r} is finer than native dataset frequency {n}; "
        f"intraday/finer bars are UNAVAILABLE (no fabricated data)"
    )


def grid_specs(
    *,
    data_timeframes: list[str] | tuple[str, ...] | None = None,
    signal_timeframes: list[str] | tuple[str, ...] | None = None,
    holding_bars: list[int] | tuple[int, ...] | None = None,
    instrument: str = "",
    strategy_id: str = "",
    strategy_version: str = "1.0.0",
) -> list[dict[str, Any]]:
    """Expand a research grid into plain dict specs (before availability gating)."""
    from iqrp.app.backtesting.horizon.types import (
        DEFAULT_DATA_TIMEFRAMES,
        DEFAULT_HOLDING_BARS,
    )

    dts = list(data_timeframes or DEFAULT_DATA_TIMEFRAMES)
    sts = list(signal_timeframes or dts)
    holds = list(holding_bars or DEFAULT_HOLDING_BARS)
    out: list[dict[str, Any]] = []
    for d in dts:
        for s in sts:
            # signal timeframe should not be finer than data timeframe
            if parse_timeframe(s).seconds + 1e-9 < parse_timeframe(d).seconds:
                continue
            for h in holds:
                out.append(
                    {
                        "data_timeframe": str(parse_timeframe(d)),
                        "signal_timeframe": str(parse_timeframe(s)),
                        "holding_bars": int(h),
                        "instrument": instrument,
                        "strategy_id": strategy_id,
                        "strategy_version": strategy_version,
                    }
                )
    return out


__all__ = [
    "availability_reason",
    "can_derive",
    "grid_specs",
    "parse_holding",
    "parse_timeframe",
    "timeframe_rank",
]
