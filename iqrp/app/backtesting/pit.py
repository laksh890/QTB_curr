"""Point-in-time (PIT) correctness helpers for institutional backtests.

CRITICAL
--------
No event handler may access data after ``event.timestamp``. These helpers
document and enforce that invariant at data-access boundaries.

Typical usage inside a handler::

    assert_no_lookahead(data_asof, event.timestamp, context="close_price")
    universe = filter_universe_asof(membership, event.timestamp)
    report = detect_leakage(feature_idx, label_idx, timestamps=index)

A backtest that fails these checks should transition to ``INVALIDATED``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class LookaheadViolation(ValueError):
    """Raised when point-in-time correctness is violated."""


def _to_comparable(ts: datetime | int | float) -> datetime | int | float:
    if isinstance(ts, datetime) and ts.tzinfo is None:
        raise LookaheadViolation(f"naive datetime is not allowed for PIT checks: {ts!r}")
    return ts


def assert_no_lookahead(
    data_timestamp: datetime | int | float,
    asof: datetime | int | float,
    *,
    context: str = "",
    allow_equal: bool = True,
) -> None:
    """Assert that ``data_timestamp`` is not strictly after ``asof``.

    Parameters
    ----------
    data_timestamp:
        Effective time of the data being read.
    asof:
        Simulation / event clock time (the latest information boundary).
    context:
        Optional label included in the error message.
    allow_equal:
        If True (default), data exactly at ``asof`` is permitted.
    """
    data_ts = _to_comparable(data_timestamp)
    asof_ts = _to_comparable(asof)
    leaked = data_ts > asof_ts if allow_equal else data_ts >= asof_ts
    if leaked:
        label = f" ({context})" if context else ""
        raise LookaheadViolation(
            f"look-ahead detected{label}: data_timestamp={data_ts!r} asof={asof_ts!r}"
        )


def filter_universe_asof(
    membership: (
        Mapping[str, Sequence[datetime | int | float]]
        | Mapping[str, Mapping[str, Any]]
        | Sequence[Mapping[str, Any]]
    ),
    asof: datetime | int | float,
    *,
    start_key: str = "start",
    end_key: str = "end",
    symbol_key: str = "symbol",
) -> list[str]:
    """Return symbols that are members of the universe at ``asof``.

    Accepted ``membership`` forms
    ------------------------------
    1. ``{symbol: (start, end)}`` — ``end`` may be ``None`` for open-ended.
    2. ``{symbol: {"start": ..., "end": ...}}``
    3. Sequence of dict rows with ``symbol`` / ``start`` / ``end`` keys.

    Assets that were not yet listed, or already delisted, at ``asof`` are
    excluded (survivorship-bias guard).
    """
    asof_ts = _to_comparable(asof)
    out: list[str] = []

    if isinstance(membership, Mapping):
        for symbol, window in membership.items():
            start, end = _extract_window(window, start_key=start_key, end_key=end_key)
            if _is_active(start, end, asof_ts):
                out.append(str(symbol))
        return out

    for row in membership:
        symbol = str(row[symbol_key])
        start = row.get(start_key)
        end = row.get(end_key)
        if start is None:
            raise LookaheadViolation(f"universe row missing {start_key}: {row!r}")
        if _is_active(_to_comparable(start), None if end is None else _to_comparable(end), asof_ts):
            out.append(symbol)
    return out


def _extract_window(
    window: Any,
    *,
    start_key: str,
    end_key: str,
) -> tuple[Any, Any]:
    if isinstance(window, Mapping):
        return window[start_key], window.get(end_key)
    if isinstance(window, Sequence) and not isinstance(window, (str, bytes)):
        if len(window) == 1:
            return window[0], None
        if len(window) >= 2:
            return window[0], window[1]
    raise LookaheadViolation(f"unsupported universe window: {window!r}")


def _is_active(start: Any, end: Any, asof: Any) -> bool:
    if asof < start:
        return False
    return not (end is not None and asof >= end)


@dataclass(slots=True)
class LeakageReport:
    """Result of :func:`detect_leakage`."""

    has_leakage: bool
    n_samples: int
    n_violations: int
    violations: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""

    def __bool__(self) -> bool:
        return self.has_leakage

    def __repr__(self) -> str:
        return (
            f"LeakageReport(has_leakage={self.has_leakage}, "
            f"n_violations={self.n_violations}/{self.n_samples}"
            f"{', ' + self.detail if self.detail else ''})"
        )


def detect_leakage(
    feature_asof_index: Sequence[int] | Sequence[float],
    label_asof_index: Sequence[int] | Sequence[float],
    *,
    timestamps: Sequence[datetime | int | float] | None = None,
    max_label_horizon: int | None = None,
) -> LeakageReport:
    """Detect look-ahead leakage between feature and label data indices.

    Design
    ------
    Treat ``feature_asof_index[i]`` as the **latest data index** consumed to
    build the feature for sample ``i``, and ``label_asof_index[i]`` as the
    **latest data index** required to compute the label for sample ``i``.

    Leakage is flagged when the label requires a strictly later index than the
    feature's as-of boundary (classic future-label / future-feature bleed)::

        label_asof_index[i] > feature_asof_index[i]

    Example (matches the platform smoke test)::

        detect_leakage([0, 1, 2], [0, 1, 3], timestamps=[1, 2, 3])
        # sample 2: feature uses index 2, label uses index 3 → leakage

    Parameters
    ----------
    feature_asof_index:
        Per-sample latest feature data index (or comparable numeric as-of).
    label_asof_index:
        Per-sample latest label data index.
    timestamps:
        Optional timeline aligned to data indices; used only for reporting.
        When provided, an out-of-range label index is also treated as leakage.
    max_label_horizon:
        Optional maximum allowed ``label_idx - feature_idx``. When set, a
        larger gap is also a violation (even if intentional forward labels
        should not enter the feature path).

    Returns
    -------
    LeakageReport
        ``has_leakage`` is True when any violation is found. The report is
        truthy iff leakage was detected.
    """
    feats = list(feature_asof_index)
    labels = list(label_asof_index)
    if len(feats) != len(labels):
        raise ValueError(f"feature/label length mismatch: {len(feats)} vs {len(labels)}")

    n = len(feats)
    violations: list[dict[str, Any]] = []
    ts = list(timestamps) if timestamps is not None else None
    max_ts_idx = (len(ts) - 1) if ts is not None else None

    for i, (f_idx, l_idx) in enumerate(zip(feats, labels)):
        f_i = int(f_idx) if not isinstance(f_idx, float) or float(f_idx).is_integer() else f_idx
        l_i = int(l_idx) if not isinstance(l_idx, float) or float(l_idx).is_integer() else l_idx

        reasons: list[str] = []
        try:
            fi_num = float(f_i)
            li_num = float(l_i)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric as-of index at sample {i}") from exc

        if li_num > fi_num:
            reasons.append(f"label index {l_i} exceeds feature as-of index {f_i}")

        if max_label_horizon is not None and (li_num - fi_num) > max_label_horizon:
            reasons.append(
                f"label horizon {li_num - fi_num} exceeds max_label_horizon={max_label_horizon}"
            )

        if max_ts_idx is not None:
            if li_num > max_ts_idx:
                reasons.append(f"label index {l_i} is past end of timestamps (max={max_ts_idx})")
            if fi_num > max_ts_idx:
                reasons.append(f"feature index {f_i} is past end of timestamps (max={max_ts_idx})")

        if reasons:
            entry: dict[str, Any] = {
                "sample": i,
                "feature_index": f_i,
                "label_index": l_i,
                "reasons": reasons,
            }
            if ts is not None and isinstance(f_i, int) and 0 <= f_i < len(ts):
                entry["feature_timestamp"] = ts[f_i]
            if ts is not None and isinstance(l_i, int) and 0 <= l_i < len(ts):
                entry["label_timestamp"] = ts[l_i]
            violations.append(entry)

    has_leakage = len(violations) > 0
    detail = ""
    if has_leakage:
        first = violations[0]
        detail = f"first_violation=sample[{first['sample']}]: " + "; ".join(first["reasons"])

    return LeakageReport(
        has_leakage=has_leakage,
        n_samples=n,
        n_violations=len(violations),
        violations=violations,
        detail=detail,
    )


def filter_frame_asof(
    timestamps: Sequence[datetime | int | float],
    asof: datetime | int | float,
) -> list[int]:
    """Return indices into ``timestamps`` with values ``<= asof`` (PIT slice)."""
    asof_ts = _to_comparable(asof)
    return [i for i, ts in enumerate(timestamps) if _to_comparable(ts) <= asof_ts]


def available_asof(
    items: Iterable[tuple[Any, datetime | int | float]],
    asof: datetime | int | float,
) -> list[Any]:
    """Filter ``(value, effective_time)`` pairs to those with ``effective_time <= asof``."""
    asof_ts = _to_comparable(asof)
    return [value for value, ts in items if _to_comparable(ts) <= asof_ts]


__all__ = [
    "LeakageReport",
    "LookaheadViolation",
    "assert_no_lookahead",
    "available_asof",
    "detect_leakage",
    "filter_frame_asof",
    "filter_universe_asof",
]
