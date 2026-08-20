"""Dataset-level corporate-action loading and normalization.

Wraps :mod:`iqrp.app.backtesting.corporate_actions` without replacing it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from iqrp.app.backtesting.corporate_actions import (
    CorporateAction,
    CorporateActionType,
    actions_asof,
    build_action,
)
from iqrp.app.backtesting.data.point_in_time import filter_frame_asof_df

__all__ = [
    "load_corporate_actions",
    "normalize_corporate_actions",
    "corporate_actions_frame",
    "corporate_actions_asof",
    "actions_to_frame",
]


_COLUMN_ALIASES = {
    "symbol": "instrument",
    "ticker": "instrument",
    "instrument": "instrument",
    "action_type": "action_type",
    "type": "action_type",
    "ca_type": "action_type",
    "ex_date": "ex_date",
    "effective_date": "ex_date",
    "date": "ex_date",
    "ratio": "ratio",
    "split_ratio": "ratio",
    "dividend": "dividend",
    "amount": "dividend",
    "new_symbol": "new_symbol",
    "action_id": "action_id",
    "id": "action_id",
}


def load_corporate_actions(
    source: str | Path | pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[CorporateAction],
) -> list[CorporateAction]:
    """Load corporate actions from CSV/Parquet/DataFrame/mappings/objects."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_parquet(path)
        return normalize_corporate_actions(frame)
    if isinstance(source, pd.DataFrame):
        return normalize_corporate_actions(source)
    if not source:
        return []
    first = next(iter(source))
    if isinstance(first, CorporateAction):
        return list(source)  # type: ignore[arg-type]
    return normalize_corporate_actions(list(source))  # type: ignore[arg-type]


def normalize_corporate_actions(
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> list[CorporateAction]:
    """Normalize heterogeneous CA rows into :class:`CorporateAction` objects."""
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
        # Lower-case normalize then coalesce aliases into canonical columns.
        lower_map = {c: str(c).strip().lower() for c in df.columns}
        df = df.rename(columns=lower_map)
        for alias, canonical in _COLUMN_ALIASES.items():
            if alias == canonical or alias not in df.columns:
                continue
            if canonical in df.columns:
                df[canonical] = df[canonical].where(df[canonical].notna(), df[alias])
                df = df.drop(columns=[alias])
            else:
                df = df.rename(columns={alias: canonical})
        if not df.columns.is_unique:
            df = df.loc[:, ~df.columns.duplicated()].copy()
        records = df.to_dict(orient="records")
    else:  # pragma: no cover - list-of-mappings path covered via load helpers
        records = []
        for row in rows:
            norm: dict[str, Any] = {}
            for k, v in row.items():
                canon = _COLUMN_ALIASES.get(str(k).strip().lower(), str(k))
                norm[canon] = v
            records.append(norm)

    actions: list[CorporateAction] = []
    for row in records:
        instrument = str(row.get("instrument", row.get("symbol", "")))
        if not instrument:
            raise ValueError(f"corporate action missing instrument: {row!r}")
        ex_raw = row.get("ex_date")
        if ex_raw is None:
            raise ValueError(f"corporate action missing ex_date: {row!r}")
        ex_ts = pd.Timestamp(ex_raw)
        if ex_ts.tzinfo is None:
            ex_ts = ex_ts.tz_localize("UTC")
        else:
            ex_ts = ex_ts.tz_convert("UTC")
        action_type = row.get("action_type", CorporateActionType.OTHER)
        payload = {
            k: v
            for k, v in row.items()
            if k
            not in {
                "instrument",
                "symbol",
                "action_type",
                "ex_date",
                "action_id",
            }
            and v is not None
            and not (isinstance(v, float) and pd.isna(v))
        }
        # Prefer ratio / dividend keys expected by parent helpers
        if "ratio" in payload and "split_ratio" not in payload:
            payload["split_ratio"] = payload["ratio"]
        action = build_action(
            action_type,
            instrument,
            ex_ts.to_pydatetime(),
            **payload,
        )
        action_id = row.get("action_id")
        if action_id:
            object.__setattr__(action, "action_id", str(action_id))
        actions.append(action)
    actions.sort(key=lambda a: (a.ex_date, a.symbol, a.action_type.value))
    return actions


def corporate_actions_frame(actions: Sequence[CorporateAction]) -> pd.DataFrame:
    """Convert actions to a normalized DataFrame."""
    return actions_to_frame(actions)


def actions_to_frame(actions: Sequence[CorporateAction]) -> pd.DataFrame:
    rows = []
    for a in actions:
        row = {
            "instrument": a.symbol,
            "action_type": a.action_type.value,
            "ex_date": a.ex_date,
            "action_id": a.action_id,
            **a.payload,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["instrument", "action_type", "ex_date", "action_id"]
        )
    df = pd.DataFrame(rows)
    df["ex_date"] = pd.to_datetime(df["ex_date"], utc=True)
    return df.sort_values(["ex_date", "instrument"]).reset_index(drop=True)


def corporate_actions_asof(
    actions: Sequence[CorporateAction] | pd.DataFrame,
    asof: datetime,
) -> list[CorporateAction]:
    """Return corporate actions with ``ex_date <= asof`` (PIT)."""
    if isinstance(actions, pd.DataFrame):
        frame = actions.copy()
        if "effective_timestamp" not in frame.columns:
            frame["effective_timestamp"] = pd.to_datetime(frame["ex_date"], utc=True)
        filtered = filter_frame_asof_df(frame, asof, timestamp_col="effective_timestamp")
        return normalize_corporate_actions(filtered)
    return actions_asof(list(actions), asof)
