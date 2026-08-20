"""Intraday bar / session validation (no silent repair)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.data.historical.calendar import (
    ExchangeCalendar,
    frequency_to_seconds,
    nse_equity_calendar,
)


@dataclass
class GapThresholds:
    minor_gap_pct: float = 2.0  # missing bars vs expected
    major_gap_pct: float = 10.0
    unusable_gap_pct: float = 40.0


@dataclass
class SessionCoverage:
    session_date: str
    expected_bars: int
    actual_bars: int
    missing_bars: int
    excess_bars: int
    missing_intervals: list[str] = field(default_factory=list)
    longest_gap_bars: int = 0
    n_gaps: int = 0
    coverage_pct: float = 100.0
    classification: str = "COMPLETE"
    calendar_explained: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date,
            "expected_bars": self.expected_bars,
            "actual_bars": self.actual_bars,
            "missing_bars": self.missing_bars,
            "excess_bars": self.excess_bars,
            "missing_intervals": list(self.missing_intervals)[:50],
            "missing_intervals_truncated": len(self.missing_intervals) > 50,
            "longest_gap_bars": self.longest_gap_bars,
            "n_gaps": self.n_gaps,
            "coverage_pct": self.coverage_pct,
            "classification": self.classification,
            "calendar_explained": self.calendar_explained,
            "notes": list(self.notes),
        }


def validate_ohlc_relationships(frame: pd.DataFrame) -> dict[str, Any]:
    """Flag invalid OHLC; do not repair."""
    o = frame["open"].to_numpy(dtype=np.float64)
    h = frame["high"].to_numpy(dtype=np.float64)
    l = frame["low"].to_numpy(dtype=np.float64)
    c = frame["close"].to_numpy(dtype=np.float64)
    high_ok = h + 1e-12 >= np.maximum(np.maximum(o, c), l)
    low_ok = l - 1e-12 <= np.minimum(np.minimum(o, c), h)
    invalid = ~(high_ok & low_ok)
    return {
        "invalid_ohlc_count": int(invalid.sum()),
        "invalid_ohlc_indices": np.where(invalid)[0][:100].tolist(),
    }


def validate_prices_volumes(frame: pd.DataFrame) -> dict[str, Any]:
    prices = frame[["open", "high", "low", "close"]]
    zero_neg_price = int(((prices <= 0) | prices.isna()).any(axis=1).sum())
    neg_vol = int((frame["volume"] < 0).sum()) if "volume" in frame.columns else 0
    return {
        "zero_or_negative_price_bars": zero_neg_price,
        "negative_volume_bars": neg_vol,
    }


def detect_duplicate_timestamps(frame: pd.DataFrame) -> dict[str, Any]:
    key = ["timestamp", "instrument"] if "instrument" in frame.columns else ["timestamp"]
    dup = frame.duplicated(key, keep=False)
    return {
        "duplicate_rows": int(dup.sum()),
        "duplicate_keys": int(frame.loc[dup, key].drop_duplicates().shape[0]) if dup.any() else 0,
    }


def detect_ordering_issues(frame: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    out_of_order = int((ts.diff().dropna() < pd.Timedelta(0)).sum())
    return {"out_of_order_steps": out_of_order, "timezone_aware": bool(getattr(ts.dt, "tz", None))}


def classify_coverage(missing_pct: float, thresholds: GapThresholds) -> str:
    if missing_pct <= 1e-9:
        return "COMPLETE"
    if missing_pct <= thresholds.minor_gap_pct:
        return "MINOR_GAPS"
    if missing_pct <= thresholds.major_gap_pct:
        return "MAJOR_GAPS"
    if missing_pct >= thresholds.unusable_gap_pct:
        return "UNUSABLE"
    return "MAJOR_GAPS"


def analyze_session_coverage(
    frame: pd.DataFrame,
    *,
    frequency: str,
    calendar: ExchangeCalendar | None = None,
    thresholds: GapThresholds | None = None,
) -> dict[str, Any]:
    """Compare expected vs actual bars per exchange session."""
    cal = calendar or nse_equity_calendar()
    thr = thresholds or GapThresholds()
    freq_sec = frequency_to_seconds(frequency)
    g = frame.copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True)
    # work in exchange tz for session dating
    local = g["timestamp"].dt.tz_convert(cal.timezone)
    g["session_date"] = local.dt.date

    sessions: list[SessionCoverage] = []
    for d, part in g.groupby("session_date"):
        d = date.fromisoformat(str(d)) if not isinstance(d, date) else d
        expected_ts = cal.expected_bar_timestamps(d, freq_sec)
        expected_set = {pd.Timestamp(t).tz_convert("UTC") for t in expected_ts}
        actual_ts = [pd.Timestamp(t).tz_convert("UTC") for t in part["timestamp"].tolist()]
        actual_set = set(actual_ts)

        if not expected_ts:
            # data on non-trading day per calendar — flag but do not invent session
            cov = SessionCoverage(
                session_date=str(d),
                expected_bars=0,
                actual_bars=len(actual_ts),
                missing_bars=0,
                excess_bars=len(actual_ts),
                coverage_pct=0.0,
                classification="MAJOR_GAPS",
                calendar_explained=True,
                notes=["bars present on day with no configured session (holiday/weekend/unknown)"],
            )
            sessions.append(cov)
            continue

        missing = sorted(expected_set - actual_set)
        excess = sorted(actual_set - expected_set)
        # gap runs in expected timeline
        present = [t in actual_set for t in sorted(expected_set)]
        longest = 0
        cur = 0
        n_gaps = 0
        for p in present:
            if not p:
                cur += 1
                longest = max(longest, cur)
            else:
                if cur > 0:
                    n_gaps += 1
                cur = 0
        if cur > 0:
            n_gaps += 1
        miss_n = len(missing)
        exp_n = len(expected_set)
        miss_pct = 100.0 * miss_n / max(exp_n, 1)
        classification = classify_coverage(miss_pct, thr)
        notes = []
        # early close / late open may explain fewer bars if configured
        if d in cal.early_closes or d in cal.late_opens or d in cal.special_sessions:
            notes.append("special session / early close / late open configured")
        sessions.append(
            SessionCoverage(
                session_date=str(d),
                expected_bars=exp_n,
                actual_bars=len(actual_set),
                missing_bars=miss_n,
                excess_bars=len(excess),
                missing_intervals=[str(t) for t in missing],
                longest_gap_bars=int(longest),
                n_gaps=int(n_gaps),
                coverage_pct=float(100.0 - miss_pct),
                classification=classification,
                calendar_explained=bool(notes),
                notes=notes,
            )
        )

    complete = sum(1 for s in sessions if s.classification == "COMPLETE")
    incomplete = len(sessions) - complete
    total_missing = sum(s.missing_bars for s in sessions)
    total_expected = sum(s.expected_bars for s in sessions)
    overall_cov = 100.0 * (1.0 - total_missing / total_expected) if total_expected else 0.0
    overall_class = classify_coverage(
        100.0 - overall_cov if total_expected else 100.0, thr
    )

    return {
        "frequency": frequency,
        "exchange_id": cal.exchange_id,
        "session_count": len(sessions),
        "complete_sessions": complete,
        "incomplete_sessions": incomplete,
        "missing_bars": total_missing,
        "expected_bars": total_expected,
        "coverage_percentage": overall_cov,
        "overall_classification": overall_class,
        "sessions": [s.to_dict() for s in sessions],
        "thresholds": {
            "minor_gap_pct": thr.minor_gap_pct,
            "major_gap_pct": thr.major_gap_pct,
            "unusable_gap_pct": thr.unusable_gap_pct,
        },
        "note": (
            "Fewer bars than a full regular session is not automatically corrupt "
            "when calendar special sessions explain it. Bars are never invented."
        ),
    }


def build_intraday_quality_report(
    frame: pd.DataFrame,
    *,
    frequency: str,
    dataset_id: str = "",
    calendar: ExchangeCalendar | None = None,
    thresholds: GapThresholds | None = None,
) -> dict[str, Any]:
    """Machine-readable data_quality payload (+ human summary fields)."""
    freq = str(frequency)
    # Daily / coarser: use existing DatasetValidator semantics; do not apply
    # intraday session bar grids to midnight daily stamps.
    if freq.lower() in {"1d", "1D", "d", "day", "daily"} or frequency_to_seconds(freq) >= 86400:
        from iqrp.app.backtesting.data.dataset_validator import DatasetValidator
        from iqrp.app.backtesting.data.metadata import DatasetMetadata

        meta = DatasetMetadata(dataset_id=dataset_id or "daily", version="1.0.0", source="local")
        report = DatasetValidator().validate(frame, metadata=meta)
        payload = report.to_dict()
        payload["frequency"] = freq
        payload["row_count"] = int(len(frame))
        payload["instrument_count"] = (
            int(frame["instrument"].nunique()) if "instrument" in frame.columns else 0
        )
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        payload["start_timestamp"] = str(ts.min()) if len(ts) else None
        payload["end_timestamp"] = str(ts.max()) if len(ts) else None
        payload["session_coverage_note"] = (
            "Intraday session expected-bar analysis skipped for daily frequency; "
            "DatasetValidator used instead."
        )
        payload["ok"] = bool(report.ok)
        payload["critical_failures"] = list(report.critical_failures)
        return payload

    ohlc = validate_ohlc_relationships(frame)
    pv = validate_prices_volumes(frame)
    dups = detect_duplicate_timestamps(frame)
    order = detect_ordering_issues(frame)
    coverage = analyze_session_coverage(
        frame, frequency=frequency, calendar=calendar, thresholds=thresholds
    )
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    price_cols = frame[["open", "high", "low", "close"]]
    nan_count = int(price_cols.isna().sum().sum())
    inf_count = int(np.isinf(price_cols.to_numpy(dtype=np.float64)).sum())
    # extreme: closes beyond 50x median absolute deviation heuristic (diagnostic only)
    closes = frame["close"].to_numpy(dtype=np.float64)
    finite = closes[np.isfinite(closes)]
    extreme = 0
    if finite.size:
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med))) or med * 0.01
        extreme = int(np.sum(np.abs(finite - med) > 50.0 * mad))
    critical = []
    if ohlc["invalid_ohlc_count"]:
        critical.append(f"invalid_ohlc={ohlc['invalid_ohlc_count']}")
    if pv["zero_or_negative_price_bars"]:
        critical.append(f"bad_prices={pv['zero_or_negative_price_bars']}")
    if pv["negative_volume_bars"]:
        critical.append(f"negative_volume={pv['negative_volume_bars']}")
    if dups["duplicate_rows"]:
        critical.append(f"duplicates={dups['duplicate_rows']}")
    if order["out_of_order_steps"]:
        critical.append(f"out_of_order={order['out_of_order_steps']}")
    if nan_count:
        critical.append(f"nan_ohlc={nan_count}")
    if inf_count:
        critical.append(f"inf_ohlc={inf_count}")
    if coverage["overall_classification"] == "UNUSABLE":
        critical.append("coverage_UNUSABLE")
    if not order["timezone_aware"]:
        critical.append("timezone_naive")

    cal = calendar or nse_equity_calendar()
    report = {
        "dataset_id": dataset_id,
        "row_count": int(len(frame)),
        "instrument_count": int(frame["instrument"].nunique()) if "instrument" in frame else 0,
        "start_timestamp": str(ts.min()) if len(ts) else None,
        "end_timestamp": str(ts.max()) if len(ts) else None,
        "frequency": frequency,
        "session_count": coverage["session_count"],
        "complete_sessions": coverage["complete_sessions"],
        "incomplete_sessions": coverage["incomplete_sessions"],
        "missing_bars": coverage["missing_bars"],
        "n_gaps": int(sum(s.get("n_gaps", 0) for s in coverage.get("sessions", []))),
        "duplicate_bars": dups["duplicate_rows"],
        "invalid_ohlc_bars": ohlc["invalid_ohlc_count"],
        "invalid_volume_bars": pv["negative_volume_bars"],
        "nan_ohlc_values": nan_count,
        "infinite_ohlc_values": inf_count,
        "extreme_value_diagnostics": {
            "n_extreme_close_vs_mad": extreme,
            "note": "Diagnostic only — not auto-repaired.",
        },
        "timezone_anomalies": 0 if order["timezone_aware"] else 1,
        "coverage_percentage": coverage["coverage_percentage"],
        "overall_gap_classification": coverage["overall_classification"],
        "market_type": getattr(cal, "market_type", "EQUITY"),
        "continuous_24x7": bool(getattr(cal, "continuous_24x7", False)),
        "ohlc": ohlc,
        "prices_volumes": pv,
        "duplicates": dups,
        "ordering": order,
        "session_coverage": coverage,
        "critical_failures": critical,
        "ok": len(critical) == 0,
        "missing_vs_zero_volume_note": (
            "Absent timestamps are missing bars; zero volume (if present) is not treated as missing."
        ),
        "lookahead_policy": (
            "No future-filled OHLC, no forward-fill of prices, "
            "no future-based repair. Observation timestamps preserved."
        ),
        "availability_timestamp_note": (
            "If provider does not supply availability timestamps, that limitation "
            "is recorded; do not pretend vendor history is perfectly point-in-time."
        ),
    }
    return report


def quality_report_markdown(report: dict[str, Any]) -> str:
    cov = report.get("coverage_percentage")
    if cov is None:
        cov_field = report.get("coverage")
        if isinstance(cov_field, dict):
            cov = cov_field.get("coverage_pct")
    cov_s = f"{float(cov):.2f}%" if cov is not None else "n/a"
    lines = [
        f"# Data Quality Report — {report.get('dataset_id', '')}",
        "",
        f"- OK: **{report.get('ok')}**",
        f"- Rows: {report.get('row_count')}",
        f"- Instruments: {report.get('instrument_count')}",
        f"- Range: {report.get('start_timestamp') or (report.get('date_range') or (None, None))[0]} "
        f"→ {report.get('end_timestamp') or (report.get('date_range') or (None, None))[1]}",
        f"- Frequency: {report.get('frequency')}",
        f"- Sessions: {report.get('session_count', 'n/a')} "
        f"(complete={report.get('complete_sessions', 'n/a')}, "
        f"incomplete={report.get('incomplete_sessions', 'n/a')})",
        f"- Missing bars: {report.get('missing_bars', 'n/a')}",
        f"- Coverage: {cov_s} "
        f"({report.get('overall_gap_classification', report.get('session_coverage_note', ''))})",
        f"- Invalid OHLC: {report.get('invalid_ohlc_bars', (report.get('invalid_data') or {}).get('ohlc', 'n/a'))}",
        f"- Duplicates: {report.get('duplicate_bars', (report.get('duplicate_data') or {}).get('count', 'n/a'))}",
        "",
        "## Critical failures",
    ]
    crit = report.get("critical_failures") or []
    if not crit:
        lines.append("- None")
    else:
        lines.extend(f"- {c}" for c in crit)
    lines.append("")
    lines.append(report.get("lookahead_policy", report.get("session_coverage_note", "")))
    return "\n".join(lines)


__all__ = [
    "GapThresholds",
    "SessionCoverage",
    "analyze_session_coverage",
    "build_intraday_quality_report",
    "detect_duplicate_timestamps",
    "detect_ordering_issues",
    "quality_report_markdown",
    "validate_ohlc_relationships",
    "validate_prices_volumes",
]
