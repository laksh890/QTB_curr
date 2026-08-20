"""Session-aware deterministic OHLCV resampling (no overnight / cross-session bars)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from iqrp.app.backtesting.data.schema import normalize_frame
from iqrp.app.data.historical.calendar import (
    ExchangeCalendar,
    frequency_to_seconds,
    nse_equity_calendar,
)
from iqrp.app.data.historical.provenance import DatasetProvenance, now_utc_iso


_PANDAS_RULE = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "60m": "1h",
}


def resample_session_aware(
    frame: pd.DataFrame,
    *,
    source_frequency: str,
    derived_frequency: str,
    calendar: ExchangeCalendar | None = None,
    source_dataset_id: str | None = None,
    source_checksum: str | None = None,
) -> tuple[pd.DataFrame, DatasetProvenance]:
    """Aggregate SOURCE bars to DERIVED frequency within each session only.

    Does not fabricate missing bars. Does not span overnight / session boundaries.
    """
    src = str(source_frequency)
    dst = str(derived_frequency)
    if frequency_to_seconds(dst) < frequency_to_seconds(src):
        raise ValueError(f"cannot derive finer {dst} from coarser source {src}")
    if abs(frequency_to_seconds(dst) - frequency_to_seconds(src)) < 1e-9:
        prov = DatasetProvenance(
            provider="derived",
            source="resample_identity",
            acquisition_timestamp=now_utc_iso(),
            original_symbol="",
            normalized_symbol="",
            frequency=dst,
            source_dataset_id=source_dataset_id,
            source_frequency=src,
            derived_frequency=dst,
            aggregation_method="identity",
            creation_timestamp=now_utc_iso(),
            frequency_kind="SOURCE",
            checksum=source_checksum or "",
        )
        return frame.copy(), prov

    rule = _PANDAS_RULE.get(dst)
    if rule is None:
        raise ValueError(f"unsupported derived frequency: {dst}")

    cal = calendar or nse_equity_calendar()
    rows: list[pd.DataFrame] = []
    g = frame.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)

    for inst, ig in g.groupby("instrument", sort=False):
        local = ig["timestamp"].dt.tz_convert(cal.timezone)
        ig = ig.assign(session_date=local.dt.date)
        for _d, sg in ig.groupby("session_date", sort=True):
            part = sg.set_index("timestamp").sort_index()
            if part.empty:
                continue
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            for opt in ("vwap", "trade_count", "open_interest", "bid", "ask"):
                if opt in part.columns:
                    agg[opt] = "last"
            # Anchor bins to the first bar of THIS session (e.g. 09:15), never midnight
            # and never across overnight / session boundaries (group is session-scoped).
            origin = part.index[0]
            r = part.resample(rule, label="left", closed="left", origin=origin).agg(agg)
            r = r.dropna(subset=["open", "high", "low", "close"])
            if r.empty:
                continue
            r = r.reset_index()
            r["instrument"] = inst
            rows.append(r)

    if not rows:
        out = frame.iloc[0:0].copy()
    else:
        out = normalize_frame(pd.concat(rows, ignore_index=True))

    creation = now_utc_iso()
    # checksum filled by caller after write
    prov = DatasetProvenance(
        provider="derived",
        source="session_aware_ohlcv_resample",
        acquisition_timestamp=creation,
        original_symbol="",
        normalized_symbol=str(frame["instrument"].iloc[0]) if len(frame) else "",
        frequency=dst,
        timezone="UTC",
        exchange_timezone=cal.timezone,
        adjustment_status="inherited",
        corporate_action_treatment="inherited",
        source_dataset_id=source_dataset_id,
        source_frequency=src,
        derived_frequency=dst,
        aggregation_method="open=first,high=max,low=min,close=last,volume=sum; session-bounded",
        creation_timestamp=creation,
        frequency_kind="DERIVED",
        known_limitations=[
            "Derived bars are aggregations of source bars; not equivalent to a native vendor feed.",
            "Missing source bars are not fabricated before aggregation.",
        ],
        extra={"source_checksum": source_checksum},
    )
    return out, prov


__all__ = ["resample_session_aware"]
