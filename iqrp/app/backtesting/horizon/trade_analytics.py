"""Trade frequency, holding-period, and long/short classification analytics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.horizon.types import SignalSide


def classify_side(side: Any) -> SignalSide:
    s = str(side or "").strip().lower()
    if s in {"buy", "b", "long", "cover", "l"}:
        return SignalSide.LONG
    if s in {"sell", "s", "short", "sell_short", "ss"}:
        return SignalSide.SHORT
    return SignalSide.FLAT


def _as_trade_rows(trades: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in trades or []:
        rows.append(dict(t))
    return rows


def holding_seconds(trade: Mapping[str, Any]) -> float | None:
    """Extract holding duration in seconds from a trade record."""
    if trade.get("holding_seconds") is not None:
        return float(trade["holding_seconds"])
    if trade.get("holding") is not None:
        try:
            return float(trade["holding"])
        except Exception:  # noqa: BLE001
            pass
    entry = trade.get("entry_time") or trade.get("entry_timestamp") or trade.get("timestamp")
    exit_ = trade.get("exit_time") or trade.get("exit_timestamp")
    if entry is None or exit_ is None:
        return None
    try:
        e = pd.Timestamp(entry)
        x = pd.Timestamp(exit_)
        if e.tzinfo is None:
            e = e.tz_localize("UTC")
        if x.tzinfo is None:
            x = x.tz_localize("UTC")
        return float((x - e).total_seconds())
    except Exception:  # noqa: BLE001
        return None


def enrich_trades_with_holding(
    trades: Sequence[Mapping[str, Any]],
    *,
    bar_seconds: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in _as_trade_rows(trades):
        sec = holding_seconds(t)
        t = dict(t)
        if sec is not None:
            t["holding_seconds"] = float(sec)
            if bar_seconds and bar_seconds > 0:
                t["holding_bars"] = float(sec) / float(bar_seconds)
        t["side_class"] = classify_side(t.get("side")).value
        out.append(t)
    return out


def trade_frequency_report(
    trades: Sequence[Mapping[str, Any]],
    *,
    timestamps: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute trade-count diagnostics (not an optimization objective)."""
    rows = enrich_trades_with_holding(trades)
    n = len(rows)
    long_n = sum(1 for t in rows if t.get("side_class") == SignalSide.LONG.value)
    short_n = sum(1 for t in rows if t.get("side_class") == SignalSide.SHORT.value)

    # Bucket by calendar day
    by_day: dict[str, int] = defaultdict(int)
    for t in rows:
        ts = t.get("exit_time") or t.get("entry_time") or t.get("timestamp")
        if ts is None:
            continue
        day = str(pd.Timestamp(ts).date())
        by_day[day] += 1

    day_counts = np.asarray(list(by_day.values()), dtype=np.float64) if by_day else np.zeros(0)
    n_days = max(len(by_day), 1)
    # Estimate weeks/months from span if timestamps provided
    span_days = float(n_days)
    if timestamps is not None:
        ts = pd.to_datetime(list(timestamps), utc=True)
        if len(ts) >= 2:
            span_days = max(float((ts.max() - ts.min()).total_seconds()) / 86400.0, 1.0)

    holdings = np.asarray(
        [float(t["holding_seconds"]) for t in rows if t.get("holding_seconds") is not None],
        dtype=np.float64,
    )

    def _pct(a: np.ndarray, q: float) -> float | None:
        return float(np.quantile(a, q)) if a.size else None

    return {
        "total_trades": n,
        "long_trades": long_n,
        "short_trades": short_n,
        "long_percentage": (long_n / n) if n else 0.0,
        "short_percentage": (short_n / n) if n else 0.0,
        "trades_per_day": n / span_days if span_days else 0.0,
        "trades_per_week": n / max(span_days / 7.0, 1e-12),
        "trades_per_month": n / max(span_days / 21.0, 1e-12),
        "average_trades_per_trading_day": float(np.mean(day_counts)) if day_counts.size else 0.0,
        "median_trades_per_day": float(np.median(day_counts)) if day_counts.size else 0.0,
        "maximum_trades_per_day": float(np.max(day_counts)) if day_counts.size else 0.0,
        "minimum_trades_per_day": float(np.min(day_counts)) if day_counts.size else 0.0,
        "n_trading_days_with_trades": int(len(by_day)),
        "average_holding_period_seconds": float(np.mean(holdings)) if holdings.size else None,
        "median_holding_period_seconds": float(np.median(holdings)) if holdings.size else None,
        "holding_period_p25_seconds": _pct(holdings, 0.25),
        "holding_period_p75_seconds": _pct(holdings, 0.75),
        "holding_period_p90_seconds": _pct(holdings, 0.90),
        "note": "Trade count is a diagnostic variable, not an optimization objective.",
    }


def side_transition_summary(signals: Sequence[Any]) -> dict[str, Any]:
    """Count LONG/SHORT/FLAT transitions in a signal path."""
    seq = [classify_side(s).value for s in signals]
    counts = {SignalSide.LONG.value: 0, SignalSide.SHORT.value: 0, SignalSide.FLAT.value: 0}
    for s in seq:
        counts[s] = counts.get(s, 0) + 1
    transitions = defaultdict(int)
    for a, b in zip(seq, seq[1:], strict=False):
        if a != b:
            transitions[f"{a}->{b}"] += 1
    return {"counts": counts, "transitions": dict(transitions), "n": len(seq)}


__all__ = [
    "classify_side",
    "enrich_trades_with_holding",
    "holding_seconds",
    "side_transition_summary",
    "trade_frequency_report",
]
