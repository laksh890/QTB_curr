"""Timestamp unit detection for Binance Vision open_time columns."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from iqrp.app.data.historical.binance_vision import BinanceVisionHistoricalProvider


def _write_zip(path: Path, open_times: list[int], *, header: bool = False) -> None:
    buf = io.StringIO()
    if header:
        buf.write("open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_base,taker_buy_quote,ignore\n")
    for t in open_times:
        buf.write(f"{t},100,101,99,100.5,1.0,{t+59999},100,10,0.5,50,0\n")
    raw = buf.getvalue().encode()
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("klines.csv", raw)


def test_parse_zip_milliseconds_2024(tmp_path: Path):
    # 2024-06-01 00:00 UTC in ms
    ms = 1717200000000
    z = tmp_path / "ms.zip"
    _write_zip(z, [ms, ms + 60_000])
    p = BinanceVisionHistoricalProvider(cache_dir=tmp_path / "cache")
    df = p._parse_zip(z, "BTCUSDT")
    assert df["timestamp"].iloc[0].year == 2024


def test_parse_zip_microseconds_2025(tmp_path: Path):
    # 2025-01-01 00:00 UTC in µs (16 digits) — previously misread as ns → 1970
    us = 1735689600000000
    z = tmp_path / "us.zip"
    _write_zip(z, [us, us + 60_000_000])
    p = BinanceVisionHistoricalProvider(cache_dir=tmp_path / "cache")
    df = p._parse_zip(z, "BTCUSDT")
    assert df["timestamp"].iloc[0].year == 2025
    assert df["timestamp"].iloc[0].month == 1
