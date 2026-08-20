"""Point-in-time helpers for historical features, signals, and universes.

Thin wrappers around :mod:`iqrp.app.backtesting.pit` with DataFrame-oriented
APIs used by the historical data package.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from iqrp.app.backtesting.pit import (
    LookaheadViolation,
    assert_no_lookahead,
    available_asof,
    filter_frame_asof,
    filter_universe_asof,
)

__all__ = [
    "LookaheadViolation",
    "assert_no_lookahead",
    "effective_timestamp",
    "ensure_effective_timestamps",
    "filter_frame_asof_df",
    "filter_features_asof",
    "filter_signals_asof",
    "filter_universe_membership_asof",
    "available_asof",
    "filter_frame_asof",
    "filter_universe_asof",
]


def effective_timestamp(
    row: Mapping[str, Any] | pd.Series,
    *,
    timestamp_key: str = "timestamp",
    effective_key: str = "effective_timestamp",
) -> datetime:
    """Return the effective (point-in-time) timestamp for an observation.

    Prefers ``effective_timestamp`` when present; otherwise uses ``timestamp``.
    Result must be timezone-aware.
    """
    if isinstance(row, pd.Series):
        data = row
        ts = data[effective_key] if effective_key in data.index and pd.notna(data[effective_key]) else data[timestamp_key]
    else:
        ts = row.get(effective_key, row.get(timestamp_key))
        if ts is None:
            ts = row[timestamp_key]
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        raise LookaheadViolation(f"naive datetime is not allowed for PIT checks: {out!r}")
    return out.to_pydatetime()


def ensure_effective_timestamps(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    effective_col: str = "effective_timestamp",
) -> pd.DataFrame:
    """Ensure ``effective_timestamp`` exists (defaults to ``timestamp``)."""
    df = frame.copy()
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")
    if effective_col not in df.columns:
        df[effective_col] = df[timestamp_col]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df[effective_col] = pd.to_datetime(df[effective_col], utc=True)
    if df[effective_col].dt.tz is None:
        raise LookaheadViolation("effective_timestamp must be timezone-aware")
    # Guard: effective cannot be after observation timestamp (future leak into past label)
    leaked = df[effective_col] > df[timestamp_col]
    if bool(leaked.any()):
        raise LookaheadViolation(
            f"effective_timestamp after timestamp in {int(leaked.sum())} row(s)"
        )
    return df


def filter_frame_asof_df(
    frame: pd.DataFrame,
    asof: datetime | pd.Timestamp,
    *,
    timestamp_col: str = "effective_timestamp",
    fallback_col: str = "timestamp",
) -> pd.DataFrame:
    """Return rows with effective timestamp ``<= asof``."""
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is None:
        raise LookaheadViolation(f"naive datetime is not allowed for PIT checks: {asof_ts!r}")
    asof_ts = asof_ts.tz_convert("UTC")
    col = timestamp_col if timestamp_col in frame.columns else fallback_col
    if col not in frame.columns:
        raise ValueError(f"missing timestamp column: {col}")
    ts = pd.to_datetime(frame[col], utc=True)
    return frame.loc[ts <= asof_ts].copy()


def filter_features_asof(
    features: pd.DataFrame,
    asof: datetime | pd.Timestamp,
    *,
    timestamp_col: str = "effective_timestamp",
) -> pd.DataFrame:
    """PIT filter for feature panels (no future feature values)."""
    return filter_frame_asof_df(features, asof, timestamp_col=timestamp_col)


def filter_signals_asof(
    signals: pd.DataFrame,
    asof: datetime | pd.Timestamp,
    *,
    timestamp_col: str = "effective_timestamp",
) -> pd.DataFrame:
    """PIT filter for signal / forecast panels."""
    return filter_frame_asof_df(signals, asof, timestamp_col=timestamp_col)


def filter_universe_membership_asof(
    membership: Mapping[str, Sequence[datetime | int | float]]
    | Mapping[str, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]]
    | pd.DataFrame,
    asof: datetime | pd.Timestamp,
    *,
    start_key: str = "start",
    end_key: str = "end",
    symbol_key: str = "instrument",
) -> list[str]:
    """PIT-aware universe membership filter.

    Accepts the same structures as :func:`filter_universe_asof`, plus a
    DataFrame with instrument/start/end columns (``symbol`` also accepted).
    """
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is None:
        raise LookaheadViolation(f"naive datetime is not allowed for PIT checks: {asof_ts!r}")
    asof_py = asof_ts.to_pydatetime()

    if isinstance(membership, pd.DataFrame):
        rows: list[dict[str, Any]] = []
        inst_col = symbol_key if symbol_key in membership.columns else (
            "symbol" if "symbol" in membership.columns else "instrument"
        )
        for _, row in membership.iterrows():
            rows.append(
                {
                    "symbol": str(row[inst_col]),
                    "start": pd.Timestamp(row[start_key]).to_pydatetime(),
                    "end": (
                        None
                        if end_key not in membership.columns or pd.isna(row[end_key])
                        else pd.Timestamp(row[end_key]).to_pydatetime()
                    ),
                }
            )
        return filter_universe_asof(rows, asof_py, symbol_key="symbol")

    # Prefer instrument key; fall back to symbol for pit helper
    try:
        return filter_universe_asof(
            membership,
            asof_py,
            start_key=start_key,
            end_key=end_key,
            symbol_key=symbol_key,
        )
    except (KeyError, LookaheadViolation):
        return filter_universe_asof(
            membership,
            asof_py,
            start_key=start_key,
            end_key=end_key,
            symbol_key="symbol",
        )
