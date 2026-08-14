"""Resample OHLCV only to equal/coarser frequencies (never fabricate finer bars)."""

from __future__ import annotations

import pandas as pd

from iqrp.app.backtesting.horizon.availability import detect_native_frequency
from iqrp.app.backtesting.horizon.parse import can_derive, parse_timeframe
from iqrp.app.backtesting.horizon.types import Timeframe


class UnavailableFrequencyError(ValueError):
    """Raised when a finer-than-native frequency is requested."""


_PANDAS_RULE: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1D": "1D",
    "1d": "1D",
}


def _rule_for(tf: Timeframe) -> str:
    if tf.label in _PANDAS_RULE:
        return _PANDAS_RULE[tf.label]
    if abs(tf.seconds - 86400.0) < 1e-6:
        return "1D"
    if tf.seconds % 3600 == 0:
        return f"{int(tf.seconds // 3600)}h"
    if tf.seconds % 60 == 0:
        return f"{int(tf.seconds // 60)}min"
    raise ValueError(f"no pandas rule for {tf}")


def resample_ohlcv(
    frame: pd.DataFrame,
    requested: Timeframe | str,
    *,
    native: Timeframe | str | None = None,
) -> pd.DataFrame:
    """Downsample (or pass-through) OHLCV to ``requested``.

    Raises :class:`UnavailableFrequencyError` if ``requested`` is finer than native.
    """
    req = parse_timeframe(requested)
    nat = parse_timeframe(native) if native is not None else detect_native_frequency(frame)
    if not can_derive(nat, req):
        raise UnavailableFrequencyError(
            f"cannot fabricate {req} from native {nat}; dataset lacks required frequency"
        )
    if abs(nat.seconds - req.seconds) < 1e-9:
        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out.sort_values(["instrument", "timestamp"]).reset_index(drop=True)

    rule = _rule_for(req)
    rows: list[pd.DataFrame] = []
    for inst, g in frame.groupby("instrument", sort=False):
        g = g.copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
        g = g.set_index("timestamp").sort_index()
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        # keep optional columns if present
        for col in ("vwap", "trade_count", "open_interest"):
            if col in g.columns:
                agg[col] = "last"
        r = g.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["close"])
        r = r.reset_index()
        r["instrument"] = inst
        rows.append(r)
    if not rows:
        return frame.iloc[0:0].copy()
    out = pd.concat(rows, ignore_index=True)
    cols = ["timestamp", "instrument", "open", "high", "low", "close", "volume"]
    extra = [c for c in out.columns if c not in cols]
    return out[cols + extra].sort_values(["instrument", "timestamp"]).reset_index(drop=True)


__all__ = ["UnavailableFrequencyError", "resample_ohlcv"]
