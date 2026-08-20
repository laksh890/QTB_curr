"""Tests for Binance Vision provider and crypto 24x7 calendar."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from iqrp.app.data.historical.binance_vision import (
    BinanceVisionHistoricalProvider,
    vision_monthly_url,
)
from iqrp.app.data.historical.calendar import crypto_24x7_calendar, frequency_to_seconds
from iqrp.app.data.historical.provider import ProviderRequest
from iqrp.app.data.historical.provider_registry import get_provider, list_providers
from iqrp.app.data.historical.resampling import resample_session_aware


def test_crypto_calendar_24x7():
    cal = crypto_24x7_calendar()
    assert cal.continuous_24x7 is True
    assert cal.market_type == "CRYPTO"
    assert cal.timezone == "UTC"
    assert cal.is_trading_day(date(2024, 1, 6))  # Saturday
    assert cal.is_trading_day(date(2024, 1, 7))  # Sunday
    bars = cal.expected_bar_timestamps(date(2024, 1, 1), 60)
    assert len(bars) == 1440
    assert bars[0].hour == 0 and bars[0].minute == 0
    assert bars[-1].hour == 23 and bars[-1].minute == 59


def test_binance_provider_registered():
    assert "binance" in list_providers()
    p = get_provider("binance")
    assert p.provider_id == "binance"
    assert p.capabilities().license_status == "UNKNOWN"


def test_vision_url():
    assert "BTCUSDT-1m-2024-01.zip" in vision_monthly_url("BTCUSDT", "1m", 2024, 1)


def test_parse_zip_fixture(tmp_path: Path):
    import io
    import zipfile

    # synthetic one-day CSV without header
    rows = []
    start_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    for i in range(10):
        t = start_ms + i * 60_000
        rows.append(f"{t},100,101,99,100.5,1.5,{t+59999},150,10,0.5,50,0")
    csv_bytes = ("\n".join(rows) + "\n").encode()
    zpath = tmp_path / "BTCUSDT-1m-2024-01.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("BTCUSDT-1m-2024-01.csv", csv_bytes)

    prov = BinanceVisionHistoricalProvider(cache_dir=tmp_path / "cache")
    df = prov._parse_zip(zpath, "BTCUSDT")
    assert len(df) == 10
    assert df["timestamp"].dt.tz is not None
    assert (df["close"] > 0).all()


def test_crypto_resample_uses_same_impl():
    cal = crypto_24x7_calendar()
    ts = pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC")
    close = 40000 + np.arange(120)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "instrument": "BTCUSDT",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
        }
    )
    out, prov = resample_session_aware(
        frame,
        source_frequency="1m",
        derived_frequency="5m",
        calendar=cal,
        source_dataset_id="btcusdt_intraday_1m@1.0.0",
    )
    assert len(out) == 24
    assert prov.frequency_kind == "DERIVED"
    assert prov.source_frequency == "1m"


def test_download_skips_404_months(tmp_path: Path):
    prov = BinanceVisionHistoricalProvider(cache_dir=tmp_path / "cache", pause_s=0.0)

    def fake_download(url: str, dest: Path):
        raise FileNotFoundError(url)

    with patch.object(prov, "_download_zip", side_effect=fake_download):
        with pytest.raises(Exception):
            prov.download(
                ProviderRequest(
                    instrument="BTCUSDT",
                    start="2099-01-01",
                    end="2099-02-01",
                    frequency="1m",
                )
            )
