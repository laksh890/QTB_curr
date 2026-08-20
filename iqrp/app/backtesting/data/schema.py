"""Canonical historical market-data schema and column normalization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "REQUIRED_COLUMNS",
    "OHLCV_COLUMNS",
    "OPTIONAL_COLUMNS",
    "ALL_KNOWN_COLUMNS",
    "COLUMN_ALIASES",
    "PRICE_COLUMNS",
    "normalize_column_names",
    "normalize_frame",
    "ensure_utc_timestamps",
    "infer_frequency",
    "validate_schema_columns",
]

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "instrument",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "adj_close",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "trade_count",
    "open_interest",
    "settlement",
    "vwap",
    "contract",
    "expiry",
    "currency",
    "exchange",
)

ALL_KNOWN_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "bid",
    "ask",
    "settlement",
    "vwap",
)

# Lower-case source name → canonical name
COLUMN_ALIASES: dict[str, str] = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "dt": "timestamp",
    "ts": "timestamp",
    "open_time": "timestamp",
    "instrument": "instrument",
    "symbol": "instrument",
    "ticker": "instrument",
    "asset": "instrument",
    "secid": "instrument",
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "volume": "volume",
    "vol": "volume",
    "v": "volume",
    "adj_close": "adj_close",
    "adjusted_close": "adj_close",
    "adjclose": "adj_close",
    "bid": "bid",
    "ask": "ask",
    "bid_size": "bid_size",
    "bidsize": "bid_size",
    "ask_size": "ask_size",
    "asksize": "ask_size",
    "trade_count": "trade_count",
    "trades": "trade_count",
    "n_trades": "trade_count",
    "open_interest": "open_interest",
    "oi": "open_interest",
    "settlement": "settlement",
    "settle": "settlement",
    "vwap": "vwap",
    "contract": "contract",
    "expiry": "expiry",
    "expiration": "expiry",
    "currency": "currency",
    "ccy": "currency",
    "exchange": "exchange",
    "venue": "exchange",
}


def normalize_column_names(columns: Sequence[str]) -> dict[str, str]:
    """Map original column names to canonical names (first-wins on collisions)."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        key = str(col).strip()
        canon = COLUMN_ALIASES.get(key.lower())
        if canon is None:
            continue
        if canon in used:  # pragma: no cover - duplicate alias columns
            continue
        mapping[key] = canon
        used.add(canon)
    return mapping


def ensure_utc_timestamps(series: pd.Series) -> pd.Series:
    """Coerce a timestamp series to timezone-aware UTC."""
    ts = pd.to_datetime(series, utc=True, errors="raise")
    if getattr(ts.dt, "tz", None) is None:  # pragma: no cover - utc=True usually sets tz
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts


def normalize_frame(frame: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """Rename aliases, coerce types, and sort by (timestamp, instrument)."""
    if frame is None:
        raise ValueError("frame is required")
    df = frame.copy() if copy else frame
    rename = normalize_column_names(list(df.columns))
    if rename:
        df = df.rename(columns=rename)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns after normalization: {missing}")

    df["timestamp"] = ensure_utc_timestamps(df["timestamp"])
    df["instrument"] = df["instrument"].astype(str)

    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            continue
        if col in ("contract", "currency", "exchange"):
            df[col] = df[col].astype(str)
        elif col == "expiry":
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["timestamp", "instrument"], kind="mergesort").reset_index(drop=True)
    return df


def validate_schema_columns(columns: Sequence[str]) -> list[str]:
    """Return list of missing required columns (after alias resolution)."""
    rename = normalize_column_names(list(columns))
    present = set(rename.values()) | {str(c) for c in columns}
    return [c for c in REQUIRED_COLUMNS if c not in present]


def infer_frequency(timestamps: Sequence[Any] | pd.Series) -> str:
    """Infer a human-readable bar frequency from sorted unique timestamps."""
    if isinstance(timestamps, pd.Series):
        uniq = pd.Series(pd.to_datetime(timestamps.unique(), utc=True)).sort_values()
    else:
        uniq = pd.Series(pd.to_datetime(list(dict.fromkeys(timestamps)), utc=True)).sort_values()
    if len(uniq) < 2:
        return "unknown"
    deltas = uniq.diff().dropna()
    if deltas.empty:
        return "unknown"
    # modal timedelta
    mode = deltas.mode()
    delta = mode.iloc[0] if not mode.empty else deltas.median()
    seconds = float(delta.total_seconds())
    mapping = {
        60.0: "1m",
        180.0: "3m",
        300.0: "5m",
        900.0: "15m",
        1800.0: "30m",
        3600.0: "1h",
        14400.0: "4h",
        86400.0: "1d",
        604800.0: "1w",
    }
    for target, label in mapping.items():
        if abs(seconds - target) <= max(1.0, target * 0.05):
            return label
    if seconds < 86400:
        minutes = seconds / 60.0
        if minutes == int(minutes):
            return f"{int(minutes)}m"
        return f"{seconds:.0f}s"  # pragma: no cover
    days = seconds / 86400.0
    if abs(days - round(days)) < 0.05:
        return f"{int(round(days))}d"
    return f"{seconds:.0f}s"  # pragma: no cover


def frame_coverage(
    frame: pd.DataFrame,
    *,
    frequency: str | None = None,
) -> dict[str, Any]:
    """Estimate per-instrument date coverage given an inferred frequency."""
    if frame.empty:
        return {"coverage_pct": 0.0, "expected_bars": 0, "observed_bars": 0}
    freq = frequency or infer_frequency(frame["timestamp"])
    # Only daily (and multiples) get calendar-gap coverage; else use observed continuity.
    observed = int(len(frame))
    if freq.endswith("d") and freq not in ("unknown",):
        try:
            n_days = int(freq[:-1])
        except ValueError:
            n_days = 1
        start = frame["timestamp"].min()
        end = frame["timestamp"].max()
        instruments = frame["instrument"].nunique()
        cal_days = max(1, int((end - start) / pd.Timedelta(days=n_days)) + 1)
        # Approximate business-day fraction for 1d
        if n_days == 1:
            start_utc = (
                start.tz_localize("UTC")
                if getattr(start, "tzinfo", None) is None
                else start.tz_convert("UTC")
            )
            end_utc = (
                end.tz_localize("UTC")
                if getattr(end, "tzinfo", None) is None
                else end.tz_convert("UTC")
            )
            expected = int(
                len(pd.bdate_range(start_utc.normalize(), end_utc.normalize()))
            )
            expected = max(expected, 1) * int(instruments)
        else:
            expected = cal_days * int(instruments)
        coverage = 100.0 * min(1.0, observed / float(expected)) if expected else 0.0
        return {
            "coverage_pct": float(coverage),
            "expected_bars": int(expected),
            "observed_bars": observed,
            "frequency": freq,
        }
    # Intra-day: coverage vs contiguous unique timestamps × instruments
    n_ts = frame["timestamp"].nunique()
    n_inst = frame["instrument"].nunique()
    expected = max(1, n_ts * n_inst)
    coverage = 100.0 * min(1.0, observed / float(expected))
    return {
        "coverage_pct": float(coverage),
        "expected_bars": int(expected),
        "observed_bars": observed,
        "frequency": freq,
    }
