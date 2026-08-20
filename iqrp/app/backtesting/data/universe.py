"""Point-in-time aware universe definitions for historical backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from iqrp.app.backtesting.data.point_in_time import filter_universe_membership_asof
from iqrp.app.backtesting.serializer import to_jsonable

__all__ = [
    "UniverseKind",
    "UniverseSpec",
    "resolve_universe",
    "single_instrument",
    "instrument_list",
    "historical_universe",
    "index_constituents",
    "futures_universe",
    "continuous_futures_universe",
    "custom_universe",
]


class UniverseKind(str, Enum):
    SINGLE = "single"
    LIST = "list"
    HISTORICAL = "historical"
    INDEX_CONSTITUENTS = "index_constituents"
    FUTURES = "futures"
    CONTINUOUS = "continuous"
    CUSTOM = "custom"


@dataclass(slots=True)
class UniverseSpec:
    """Declarative universe configuration (PIT-aware when membership provided)."""

    kind: UniverseKind
    instruments: list[str] = field(default_factory=list)
    membership: list[dict[str, Any]] = field(default_factory=list)
    root: str | None = None
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, UniverseKind):
            object.__setattr__(self, "kind", UniverseKind(str(self.kind)))
        object.__setattr__(self, "instruments", [str(i) for i in self.instruments])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return to_jsonable(d)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UniverseSpec:
        payload = dict(data)
        kind = UniverseKind(str(payload.pop("kind")))
        return cls(kind=kind, **{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})

    def asof(self, when: datetime | pd.Timestamp) -> list[str]:
        """Resolve membership at ``when`` (PIT)."""
        return resolve_universe(self, asof=when)


def single_instrument(instrument: str, *, name: str | None = None) -> UniverseSpec:
    return UniverseSpec(
        kind=UniverseKind.SINGLE,
        instruments=[str(instrument)],
        name=name or str(instrument),
    )


def instrument_list(instruments: Sequence[str], *, name: str | None = None) -> UniverseSpec:
    return UniverseSpec(
        kind=UniverseKind.LIST,
        instruments=[str(i) for i in instruments],
        name=name or "list",
    )


def historical_universe(
    membership: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    name: str | None = None,
) -> UniverseSpec:
    """Universe defined by historical membership windows.

    Each row / mapping should include ``instrument`` (or ``symbol``), ``start``,
    and optional ``end``.
    """
    rows = _membership_rows(membership)
    return UniverseSpec(
        kind=UniverseKind.HISTORICAL,
        membership=rows,
        instruments=sorted({r["instrument"] for r in rows}),
        name=name or "historical",
    )


def index_constituents(
    membership: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    index_name: str,
) -> UniverseSpec:
    """Index-constituent universe (PIT via membership history)."""
    rows = _membership_rows(membership)
    return UniverseSpec(
        kind=UniverseKind.INDEX_CONSTITUENTS,
        membership=rows,
        instruments=sorted({r["instrument"] for r in rows}),
        name=index_name,
        params={"index_name": index_name},
    )


def futures_universe(
    contracts: Sequence[str],
    *,
    root: str | None = None,
    name: str | None = None,
) -> UniverseSpec:
    """Static list of futures contract identifiers (raw series)."""
    return UniverseSpec(
        kind=UniverseKind.FUTURES,
        instruments=[str(c) for c in contracts],
        root=root,
        name=name or (root or "futures"),
        params={"series_kind": "raw"},
    )


def continuous_futures_universe(
    continuous_symbol: str,
    *,
    root: str | None = None,
    name: str | None = None,
    tradable: bool = False,
) -> UniverseSpec:
    """Continuous futures research or tradable continuous symbol.

    ``tradable=False`` marks a continuous *research* series; ``tradable=True``
    indicates a front-month / tradable continuous identifier.
    """
    return UniverseSpec(
        kind=UniverseKind.CONTINUOUS,
        instruments=[str(continuous_symbol)],
        root=root,
        name=name or str(continuous_symbol),
        params={
            "series_kind": "tradable" if tradable else "continuous_research",
            "tradable": bool(tradable),
        },
    )


def custom_universe(
    instruments: Sequence[str] | None = None,
    *,
    membership: Sequence[Mapping[str, Any]] | pd.DataFrame | None = None,
    resolver: str | None = None,
    name: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> UniverseSpec:
    rows = _membership_rows(membership) if membership is not None else []
    inst = [str(i) for i in (instruments or [])]
    if not inst and rows:
        inst = sorted({r["instrument"] for r in rows})
    return UniverseSpec(
        kind=UniverseKind.CUSTOM,
        instruments=inst,
        membership=rows,
        name=name or "custom",
        params={"resolver": resolver, **dict(params or {})},
    )


def resolve_universe(
    spec: UniverseSpec | Mapping[str, Any],
    *,
    asof: datetime | pd.Timestamp | None = None,
    custom_resolvers: Mapping[str, Callable[..., list[str]]] | None = None,
) -> list[str]:
    """Resolve a universe spec to instrument ids, optionally at ``asof``."""
    if not isinstance(spec, UniverseSpec):
        spec = UniverseSpec.from_dict(spec)

    if spec.kind in (UniverseKind.SINGLE, UniverseKind.LIST, UniverseKind.FUTURES):
        return list(spec.instruments)

    if spec.kind is UniverseKind.CONTINUOUS:
        return list(spec.instruments)

    if spec.kind in (
        UniverseKind.HISTORICAL,
        UniverseKind.INDEX_CONSTITUENTS,
    ):
        if asof is None:
            return list(spec.instruments)
        return filter_universe_membership_asof(
            _hydrate_membership(spec.membership), asof
        )

    if spec.kind is UniverseKind.CUSTOM:
        resolver_name = (spec.params or {}).get("resolver")
        if resolver_name and custom_resolvers and resolver_name in custom_resolvers:
            return list(custom_resolvers[resolver_name](spec, asof=asof))
        if spec.membership and asof is not None:
            return filter_universe_membership_asof(
                _hydrate_membership(spec.membership), asof
            )
        return list(spec.instruments)

    raise ValueError(f"unsupported universe kind: {spec.kind}")


def _hydrate_membership(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert ISO start/end strings back to timezone-aware datetimes for PIT."""
    out: list[dict[str, Any]] = []
    for row in rows:
        start = row["start"]
        end = row.get("end")
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        end_py = None
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            end_py = end_ts.to_pydatetime()
        out.append(
            {
                "instrument": str(row.get("instrument", row.get("symbol", ""))),
                "symbol": str(row.get("instrument", row.get("symbol", ""))),
                "start": start_ts.to_pydatetime(),
                "end": end_py,
            }
        )
    return out


def _membership_rows(
    membership: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    if isinstance(membership, pd.DataFrame):
        inst_col = (
            "instrument"
            if "instrument" in membership.columns
            else "symbol"
            if "symbol" in membership.columns
            else None
        )
        if inst_col is None or "start" not in membership.columns:
            raise ValueError("membership DataFrame requires instrument/symbol and start")
        rows: list[dict[str, Any]] = []
        for _, row in membership.iterrows():
            end = None
            if "end" in membership.columns and pd.notna(row["end"]):
                end = pd.Timestamp(row["end"]).isoformat()
            rows.append(
                {
                    "instrument": str(row[inst_col]),
                    "symbol": str(row[inst_col]),
                    "start": pd.Timestamp(row["start"]).isoformat(),
                    "end": end,
                }
            )
        return rows

    out: list[dict[str, Any]] = []
    for row in membership:
        inst = str(row.get("instrument", row.get("symbol", "")))
        if not inst:
            raise ValueError(f"membership row missing instrument/symbol: {row!r}")
        start = row.get("start")
        if start is None:
            raise ValueError(f"membership row missing start: {row!r}")
        end = row.get("end")
        out.append(
            {
                "instrument": inst,
                "symbol": inst,
                "start": pd.Timestamp(start).isoformat(),
                "end": None if end is None or (isinstance(end, float) and pd.isna(end)) else pd.Timestamp(end).isoformat(),
            }
        )
    return out
