"""Unit tests for historical intraday acquisition primitives."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.data.dataset_registry import DatasetRegistry, compute_checksum
from iqrp.app.data.historical.calendar import (
    SessionSpec,
    frequency_to_seconds,
    nse_equity_calendar,
)
from iqrp.app.data.historical.intraday_validation import (
    analyze_session_coverage,
    build_intraday_quality_report,
    detect_duplicate_timestamps,
    validate_ohlc_relationships,
)
from iqrp.app.data.historical.provenance import DatasetProvenance
from iqrp.app.data.historical.registry_ops import (
    DatasetImmutabilityError,
    register_immutable,
)
from iqrp.app.data.historical.resampling import resample_session_aware
from iqrp.app.data.historical.timestamps import NaiveTimestampError, ensure_aware_utc


def _session_frame(n_days: int = 1, freq: str = "1min", seed: int = 1) -> pd.DataFrame:
    cal = nse_equity_calendar()
    # use a known weekday
    d0 = date(2026, 8, 10)  # Monday
    rows = []
    rng = np.random.default_rng(seed)
    px = 25000.0
    for i in range(n_days):
        d = date.fromordinal(d0.toordinal() + i)
        while not cal.is_trading_day(d):
            d = date.fromordinal(d.toordinal() + 1)
        stamps = cal.expected_bar_timestamps(d, frequency_to_seconds("1m"))
        for ts in stamps:
            shock = float(rng.normal(0, 1.0))
            o = px
            c = px + shock
            h = max(o, c) + abs(shock) * 0.1
            l = min(o, c) - abs(shock) * 0.1
            rows.append(
                {
                    "timestamp": pd.Timestamp(ts).tz_convert("UTC"),
                    "instrument": "NIFTY50",
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": 1000.0,
                }
            )
            px = c
        d0 = d
    return pd.DataFrame(rows)


def test_ensure_aware_utc_rejects_naive_without_tz():
    s = pd.Series(pd.date_range("2020-01-01", periods=3, freq="1h"))
    with pytest.raises(NaiveTimestampError):
        ensure_aware_utc(s)


def test_ensure_aware_utc_with_explicit_tz():
    s = pd.Series(pd.date_range("2020-01-01 09:15", periods=3, freq="1min"))
    rec = {}
    out = ensure_aware_utc(s, assume_timezone="Asia/Kolkata", record=rec)
    assert str(out.dt.tz) == "UTC"
    assert "localized" in rec["timestamp_conversion"]


def test_ohlc_validation_flags_bad_bars():
    df = _session_frame(1)
    bad = df.iloc[:1].copy()
    bad.loc[:, "high"] = bad["low"] - 1
    rep = validate_ohlc_relationships(pd.concat([df, bad], ignore_index=True))
    assert rep["invalid_ohlc_count"] >= 1


def test_duplicate_detection():
    df = _session_frame(1)
    dup = pd.concat([df.iloc[:2], df.iloc[:2]], ignore_index=True)
    assert detect_duplicate_timestamps(dup)["duplicate_rows"] >= 2


def test_session_expected_bars_1m():
    cal = nse_equity_calendar()
    d = date(2026, 8, 10)
    bars = cal.expected_bar_timestamps(d, 60)
    assert len(bars) == 375  # 09:15–15:30
    assert bars[0].hour == 9 and bars[0].minute == 15
    assert bars[-1].hour == 15 and bars[-1].minute == 29


def test_session_coverage_complete_fixture():
    df = _session_frame(1)
    cov = analyze_session_coverage(df, frequency="1m")
    assert cov["complete_sessions"] >= 1
    assert cov["overall_classification"] == "COMPLETE"


def test_missing_bar_detection():
    df = _session_frame(1)
    df2 = df.iloc[10:].reset_index(drop=True)
    cov = analyze_session_coverage(df2, frequency="1m")
    assert cov["missing_bars"] >= 10


def test_resample_session_aware_and_provenance(tmp_path: Path):
    df = _session_frame(1)
    out, prov = resample_session_aware(
        df,
        source_frequency="1m",
        derived_frequency="5m",
        source_dataset_id="demo@1.0.0",
    )
    assert len(out) < len(df)
    assert prov.frequency_kind == "DERIVED"
    assert prov.source_frequency == "1m"
    assert prov.derived_frequency == "5m"
    assert "session-bounded" in (prov.aggregation_method or "")


def test_checksum_changes_with_data(tmp_path: Path):
    df = _session_frame(1)
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    df.to_parquet(p1, index=False)
    df2 = df.copy()
    df2.loc[0, "close"] = float(df2.loc[0, "close"]) + 1.0
    df2.to_parquet(p2, index=False)
    assert compute_checksum(p1) != compute_checksum(p2)


def test_immutable_registration(tmp_path: Path):
    df = _session_frame(1)
    path = tmp_path / "nifty50_intraday_1m.parquet"
    df.to_parquet(path, index=False)
    reg = DatasetRegistry(tmp_path / "reg.json")
    prov = DatasetProvenance(
        provider="test",
        source="fixture",
        acquisition_timestamp="2026-01-01T00:00:00+00:00",
        original_symbol="^NSEI",
        normalized_symbol="NIFTY50",
        frequency="1m",
        data_class="DEVELOPMENT DATA",
        license_status="UNKNOWN",
    )
    register_immutable(
        reg,
        path=path,
        dataset_id="nifty50_intraday_1m",
        version="1.0.0",
        source="fixture",
        frame=df,
        provenance=prov,
        quality_status="PASS",
    )
    with pytest.raises(DatasetImmutabilityError):
        register_immutable(
            reg,
            path=path,
            dataset_id="nifty50_intraday_1m",
            version="1.0.0",
            source="fixture",
            frame=df,
            provenance=prov,
        )


def test_resample_30m_1h_session_anchored_complete():
    df = _session_frame(1)
    for freq in ("30m", "1h"):
        out, prov = resample_session_aware(
            df, source_frequency="1m", derived_frequency=freq, source_dataset_id="x"
        )
        cov = analyze_session_coverage(out, frequency=freq)
        assert cov["overall_classification"] == "COMPLETE", (freq, cov)
        assert prov.frequency_kind == "DERIVED"


def test_early_close_session_not_auto_corrupt():
    cal = nse_equity_calendar(
        early_closes={date(2026, 8, 10): time(13, 0)},
    )
    stamps = cal.expected_bar_timestamps(date(2026, 8, 10), 60)
    assert len(stamps) < 375
    assert stamps[-1].hour == 12 and stamps[-1].minute == 59
