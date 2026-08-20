"""Strict historical dataset validation (no silent repair of critical issues)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from iqrp.app.backtesting.data.metadata import DatasetMetadata
from iqrp.app.backtesting.data.schema import (
    OHLCV_COLUMNS,
    PRICE_COLUMNS,
    REQUIRED_COLUMNS,
    frame_coverage,
    infer_frequency,
    normalize_column_names,
    normalize_frame,
)
from iqrp.app.backtesting.serializer import to_jsonable

__all__ = [
    "ValidationIssue",
    "DataQualityReport",
    "DatasetValidator",
    "ValidationError",
]


class ValidationError(ValueError):
    """Raised when critical data-quality requirements are violated."""

    def __init__(self, message: str, report: DataQualityReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(slots=True)
class ValidationIssue:
    """A single validation finding."""

    code: str
    message: str
    severity: str = "warning"  # critical | warning | info
    count: int = 1
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class DataQualityReport:
    """Institutional data-quality report for a historical dataset.

    Critical violations set ``ok`` to False and populate ``critical_failures``.
    Callers must not silently repair critical problems — fail the backtest.
    """

    dataset_id: str = "unnamed"
    dataset_version: str = "1.0.0"
    source: str = "local"
    date_range: tuple[str | None, str | None] = (None, None)
    frequency: str = "unknown"
    instrument_count: int = 0
    row_count: int = 0
    missing_data: dict[str, Any] = field(default_factory=dict)
    duplicate_data: dict[str, Any] = field(default_factory=dict)
    invalid_data: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    timezone: str = "UTC"
    corporate_action_availability: bool = False
    liquidity_data_availability: bool = False
    known_limitations: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    critical_failures: list[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["issues"] = [i.to_dict() if isinstance(i, ValidationIssue) else i for i in self.issues]
        return to_jsonable(d)

    def raise_if_critical(self) -> None:
        if self.critical_failures:
            raise ValidationError(
                "critical data-quality failures: " + "; ".join(self.critical_failures),
                report=self,
            )


class DatasetValidator:
    """Validate schema, dtypes, ordering, duplicates, OHLC integrity, and timezone."""

    def __init__(
        self,
        *,
        allow_missing_optional: bool = True,
        max_missing_pct: float = 5.0,
        require_sorted: bool = True,
        fail_on_duplicates: bool = True,
        fail_on_invalid_ohlc: bool = True,
        fail_on_negative_prices: bool = True,
        fail_on_negative_volume: bool = True,
        fail_on_naive_timestamps: bool = True,
        fail_on_missing_required: bool = True,
    ) -> None:
        self.allow_missing_optional = allow_missing_optional
        self.max_missing_pct = float(max_missing_pct)
        self.require_sorted = require_sorted
        self.fail_on_duplicates = fail_on_duplicates
        self.fail_on_invalid_ohlc = fail_on_invalid_ohlc
        self.fail_on_negative_prices = fail_on_negative_prices
        self.fail_on_negative_volume = fail_on_negative_volume
        self.fail_on_naive_timestamps = fail_on_naive_timestamps
        self.fail_on_missing_required = fail_on_missing_required

    def validate(
        self,
        frame: pd.DataFrame,
        *,
        metadata: DatasetMetadata | Mapping[str, Any] | None = None,
        normalize: bool = True,
        raise_on_critical: bool = False,
    ) -> DataQualityReport:
        meta = self._coerce_metadata(metadata)
        issues: list[ValidationIssue] = []
        critical: list[str] = []

        if frame is None:
            report = DataQualityReport(
                dataset_id=meta.dataset_id,
                dataset_version=meta.version,
                source=meta.source,
                ok=False,
                critical_failures=["frame is None"],
                issues=[
                    ValidationIssue(
                        code="empty_frame",
                        message="frame is None",
                        severity="critical",
                    )
                ],
            )
            if raise_on_critical:
                report.raise_if_critical()
            return report

        # Schema (pre-normalize) — detect missing required via aliases
        rename = normalize_column_names(list(frame.columns))
        present = set(rename.values()) | {str(c) for c in frame.columns}
        missing_required = [c for c in REQUIRED_COLUMNS if c not in present]
        if missing_required:
            msg = f"missing required columns: {missing_required}"
            issues.append(
                ValidationIssue(
                    code="schema",
                    message=msg,
                    severity="critical",
                    details={"missing": missing_required},
                )
            )
            critical.append(msg)
            report = self._build_report(
                frame=frame,
                meta=meta,
                issues=issues,
                critical=critical,
                frequency="unknown",
            )
            if raise_on_critical:
                report.raise_if_critical()
            return report

        df = normalize_frame(frame) if normalize else frame

        # Timezone / invalid timestamps
        ts = df["timestamp"]
        if getattr(ts.dt, "tz", None) is None:
            msg = "timestamps are timezone-naive; UTC-aware required"
            issues.append(
                ValidationIssue(code="timezone", message=msg, severity="critical")
            )
            if self.fail_on_naive_timestamps:
                critical.append(msg)
        else:
            if str(ts.dt.tz) not in ("UTC", "UTC+00:00"):
                issues.append(
                    ValidationIssue(
                        code="timezone",
                        message=f"timestamps not UTC (found {ts.dt.tz})",
                        severity="warning",
                    )
                )

        if ts.isna().any():
            n = int(ts.isna().sum())
            msg = f"invalid/missing timestamps: {n}"
            issues.append(
                ValidationIssue(
                    code="invalid_timestamp",
                    message=msg,
                    severity="critical",
                    count=n,
                )
            )
            critical.append(msg)

        # Dtypes for OHLCV
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                msg = f"column {col!r} is not numeric"
                issues.append(
                    ValidationIssue(code="dtype", message=msg, severity="critical")
                )
                critical.append(msg)

        # Ordering
        if self.require_sorted and len(df) > 1:
            ordered = df[["timestamp", "instrument"]].apply(tuple, axis=1)
            if not ordered.is_monotonic_increasing:
                # Check strict timestamp order within each instrument
                bad = 0
                for _, grp in df.groupby("instrument", sort=False):
                    if not grp["timestamp"].is_monotonic_increasing:
                        bad += 1
                if bad:
                    msg = f"timestamp ordering violations in {bad} instrument(s)"
                    issues.append(
                        ValidationIssue(
                            code="ordering",
                            message=msg,
                            severity="critical",
                            count=bad,
                        )
                    )
                    critical.append(msg)

        # Duplicates: (timestamp, instrument)
        dup_mask = df.duplicated(subset=["timestamp", "instrument"], keep=False)
        n_dup = int(dup_mask.sum())
        duplicate_data = {
            "duplicate_rows": n_dup,
            "duplicate_keys": int(
                df.loc[dup_mask].groupby(["timestamp", "instrument"]).ngroups
            )
            if n_dup
            else 0,
        }
        if n_dup:
            msg = f"duplicate (timestamp, instrument) rows: {n_dup}"
            sev = "critical" if self.fail_on_duplicates else "warning"
            issues.append(
                ValidationIssue(
                    code="duplicates",
                    message=msg,
                    severity=sev,
                    count=n_dup,
                )
            )
            if self.fail_on_duplicates:
                critical.append(msg)

        # Missing values in required columns
        missing_counts = {
            c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS if c in df.columns
        }
        total_cells = max(1, len(df) * len(REQUIRED_COLUMNS))
        missing_cells = sum(missing_counts.values())
        missing_pct = 100.0 * missing_cells / total_cells
        missing_data = {
            "per_column": missing_counts,
            "missing_cells": missing_cells,
            "missing_pct": float(missing_pct),
        }
        if missing_cells and self.fail_on_missing_required:
            msg = f"missing required values: {missing_counts}"
            issues.append(
                ValidationIssue(
                    code="missing",
                    message=msg,
                    severity="critical",
                    count=missing_cells,
                    details=missing_counts,
                )
            )
            critical.append(msg)
        elif missing_pct > self.max_missing_pct:
            msg = f"missing_pct {missing_pct:.2f} exceeds max {self.max_missing_pct}"
            issues.append(
                ValidationIssue(code="missing", message=msg, severity="warning")
            )

        # Missing dates (gaps) — informational / warning, not always critical
        freq = infer_frequency(df["timestamp"])
        gap_count = self._count_gaps(df, freq)
        if gap_count:
            issues.append(
                ValidationIssue(
                    code="missing_dates",
                    message=f"detected {gap_count} intra-series gaps",
                    severity="warning",
                    count=gap_count,
                )
            )

        # OHLC validity
        invalid = self._invalid_ohlc_mask(df)
        n_invalid = int(invalid.sum()) if len(invalid) else 0
        invalid_data: dict[str, Any] = {"invalid_ohlc": n_invalid}

        # Negative prices
        neg_price = 0
        for col in PRICE_COLUMNS:
            if col not in df.columns:
                continue
            n = int((df[col].dropna() < 0).sum())
            if n:
                neg_price += n
                invalid_data[f"negative_{col}"] = n
        if neg_price:
            msg = f"negative prices: {neg_price}"
            sev = "critical" if self.fail_on_negative_prices else "warning"
            issues.append(
                ValidationIssue(
                    code="negative_price",
                    message=msg,
                    severity=sev,
                    count=neg_price,
                )
            )
            if self.fail_on_negative_prices:
                critical.append(msg)

        # Negative volume
        if "volume" in df.columns:
            n_neg_vol = int((df["volume"].dropna() < 0).sum())
            invalid_data["negative_volume"] = n_neg_vol
            if n_neg_vol:
                msg = f"negative volume: {n_neg_vol}"
                sev = "critical" if self.fail_on_negative_volume else "warning"
                issues.append(
                    ValidationIssue(
                        code="negative_volume",
                        message=msg,
                        severity=sev,
                        count=n_neg_vol,
                    )
                )
                if self.fail_on_negative_volume:
                    critical.append(msg)

        if n_invalid:
            msg = f"invalid OHLC relationships: {n_invalid}"
            sev = "critical" if self.fail_on_invalid_ohlc else "warning"
            issues.append(
                ValidationIssue(
                    code="invalid_ohlc",
                    message=msg,
                    severity=sev,
                    count=n_invalid,
                )
            )
            if self.fail_on_invalid_ohlc:
                critical.append(msg)

        # Zero / non-positive prices (warning)
        for col in ("open", "high", "low", "close"):
            if col in df.columns:
                n_zero = int((df[col].dropna() <= 0).sum())
                if n_zero:
                    invalid_data[f"non_positive_{col}"] = n_zero
                    issues.append(
                        ValidationIssue(
                            code="non_positive_price",
                            message=f"non-positive {col}: {n_zero}",
                            severity="warning",
                            count=n_zero,
                        )
                    )

        coverage = frame_coverage(df, frequency=freq)
        coverage["gap_count"] = gap_count

        liq = bool({"bid", "ask", "bid_size", "ask_size"}.intersection(df.columns)) or (
            "volume" in df.columns and bool(df["volume"].notna().any())
        )

        report = self._build_report(
            frame=df,
            meta=meta,
            issues=issues,
            critical=critical,
            frequency=freq,
            missing_data=missing_data,
            duplicate_data=duplicate_data,
            invalid_data=invalid_data,
            coverage=coverage,
            liquidity=liq,
        )
        if raise_on_critical:
            report.raise_if_critical()
        return report

    @staticmethod
    def _invalid_ohlc_mask(df: pd.DataFrame) -> pd.Series:
        if not all(c in df.columns for c in ("open", "high", "low", "close")):
            return pd.Series(False, index=df.index)
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]
        return (h < l) | (h < o) | (h < c) | (l > o) | (l > c)

    @staticmethod
    def _count_gaps(df: pd.DataFrame, frequency: str) -> int:
        if df.empty or frequency in ("unknown",):
            return 0
        # Approximate expected step from frequency label
        step = DatasetValidator._freq_to_timedelta(frequency)
        if step is None:
            return 0
        # Daily bars are typically business-day spaced; allow weekends/holidays
        # before counting a true gap (threshold ≈ 1.5× for intraday, 4d for 1d).
        if frequency == "1d":
            threshold = pd.Timedelta(days=4)
        else:
            threshold = step * 1.5
        gaps = 0
        for _, grp in df.groupby("instrument", sort=False):
            ts = grp["timestamp"].sort_values()
            if len(ts) < 2:
                continue
            deltas = ts.diff().dropna()
            gaps += int((deltas > threshold).sum())
        return gaps

    @staticmethod
    def _freq_to_timedelta(frequency: str) -> pd.Timedelta | None:
        try:
            if frequency.endswith("m") and frequency[:-1].isdigit():
                return pd.Timedelta(minutes=int(frequency[:-1]))
            if frequency.endswith("h") and frequency[:-1].isdigit():
                return pd.Timedelta(hours=int(frequency[:-1]))
            if frequency.endswith("d") and frequency[:-1].isdigit():
                return pd.Timedelta(days=int(frequency[:-1]))
            if frequency.endswith("w") and frequency[:-1].isdigit():
                return pd.Timedelta(weeks=int(frequency[:-1]))
            if frequency.endswith("s") and frequency[:-1].isdigit():
                return pd.Timedelta(seconds=int(frequency[:-1]))
        except (ValueError, TypeError):
            return None
        return None

    @staticmethod
    def _coerce_metadata(
        metadata: DatasetMetadata | Mapping[str, Any] | None,
    ) -> DatasetMetadata:
        if metadata is None:
            return DatasetMetadata(dataset_id="unnamed")
        if isinstance(metadata, DatasetMetadata):
            return metadata
        return DatasetMetadata.from_dict(metadata)

    def _build_report(
        self,
        *,
        frame: pd.DataFrame,
        meta: DatasetMetadata,
        issues: list[ValidationIssue],
        critical: list[str],
        frequency: str,
        missing_data: dict[str, Any] | None = None,
        duplicate_data: dict[str, Any] | None = None,
        invalid_data: dict[str, Any] | None = None,
        coverage: dict[str, Any] | None = None,
        liquidity: bool | None = None,
    ) -> DataQualityReport:
        start = end = None
        instruments = 0
        rows = int(len(frame)) if frame is not None else 0
        if frame is not None and not frame.empty and "timestamp" in frame.columns:
            try:
                start = str(pd.Timestamp(frame["timestamp"].min()))
                end = str(pd.Timestamp(frame["timestamp"].max()))
            except Exception:  # noqa: BLE001
                start = end = None
            if "instrument" in frame.columns:
                instruments = int(frame["instrument"].nunique())

        limitations = list(meta.known_limitations)
        for issue in issues:
            if issue.severity == "warning":
                limitations.append(issue.message)

        return DataQualityReport(
            dataset_id=meta.dataset_id,
            dataset_version=meta.version,
            source=meta.source,
            date_range=(start, end),
            frequency=frequency or meta.frequency,
            instrument_count=instruments or meta.instrument_count,
            row_count=rows or meta.row_count,
            missing_data=missing_data or {},
            duplicate_data=duplicate_data or {},
            invalid_data=invalid_data or {},
            coverage=coverage or {},
            timezone=meta.timezone or "UTC",
            corporate_action_availability=meta.corporate_actions_available,
            liquidity_data_availability=(
                bool(liquidity)
                if liquidity is not None
                else meta.liquidity_data_available
            ),
            known_limitations=list(dict.fromkeys(limitations)),
            issues=issues,
            critical_failures=list(dict.fromkeys(critical)),
            ok=len(critical) == 0,
        )
