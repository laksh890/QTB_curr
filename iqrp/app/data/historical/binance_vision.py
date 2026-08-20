"""Binance public historical Vision archive provider (no API key).

Source: https://data.binance.vision/ — spot monthly kline ZIPs.

DEVELOPMENT/RESEARCH data tier — not institutional-grade.
license_status = UNKNOWN (do not invent licensing claims).
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from iqrp.app.backtesting.data.schema import normalize_frame
from iqrp.app.data.historical.provider import (
    EmptyResponseError,
    HistoricalDataProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    RateLimitError,
)
from iqrp.app.data.historical.provenance import data_class_label, now_utc_iso

VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"
SUPPORTED_INTERVALS = ("1m", "5m", "15m", "30m", "1h")

# Binance kline CSV may or may not include a header depending on vintage.
_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")
    if e.tzinfo is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")
    cur = pd.Timestamp(year=s.year, month=s.month, day=1, tz="UTC")
    last = pd.Timestamp(year=e.year, month=e.month, day=1, tz="UTC")
    out: list[pd.Timestamp] = []
    while cur <= last:
        out.append(cur)
        # advance one calendar month
        y, m = cur.year, cur.month + 1
        if m > 12:
            y, m = y + 1, 1
        cur = pd.Timestamp(year=y, month=m, day=1, tz="UTC")
    return out


def vision_monthly_url(symbol: str, interval: str, year: int, month: int) -> str:
    sym = symbol.upper()
    return f"{VISION_BASE}/{sym}/{interval}/{sym}-{interval}-{year:04d}-{month:02d}.zip"


class BinanceVisionHistoricalProvider(HistoricalDataProvider):
    """Download Binance Vision monthly kline archives into canonical OHLCV."""

    provider_id = "binance"

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        pause_s: float = 0.35,
        user_agent: str = "iqrp-historical-data/0.1 (research)",
    ) -> None:
        self.cache_dir = Path(cache_dir or "data/cache/binance_vision")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.pause_s = float(pause_s)
        self.user_agent = user_agent

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.provider_id,
            supported_frequencies=SUPPORTED_INTERVALS,
            data_class="DEVELOPMENT/RESEARCH",
            rate_limits={
                "pause_seconds": self.pause_s,
                "max_retries": self.max_retries,
                "note": "Public Vision CDN; polite sequential monthly downloads.",
            },
            notes=(
                "Binance public historical Vision archives (no API key).",
                "DEVELOPMENT/RESEARCH data tier — not institutional-grade.",
                "license_status=UNKNOWN unless independently established.",
                "Availability timestamps not provided by archive CSVs.",
                "Market is 24x7 UTC continuous.",
            ),
            license_status="UNKNOWN",
        )

    def list_instruments(self) -> list[str]:
        return ["BTCUSDT"]

    def supported_frequencies(self) -> tuple[str, ...]:
        return SUPPORTED_INTERVALS

    def rate_limits(self) -> dict[str, Any]:
        return dict(self.capabilities().rate_limits)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "source": "binance_vision",
            "base_url": VISION_BASE,
            "data_class": "DEVELOPMENT/RESEARCH",
            "license_status": "UNKNOWN",
            "market_type": "CRYPTO",
            "continuous_market": True,
            "timezone": "UTC",
            "session_model": "24x7",
            "api_key_required": False,
        }

    def available_history(self, instrument: str, frequency: str) -> dict[str, Any]:
        return {
            "instrument": instrument,
            "frequency": frequency,
            "approximate_start": "2017-08 (spot BTCUSDT typically)",
            "note": "Exact availability depends on Vision archive presence per month.",
        }

    def _download_zip(self, url: str, dest: Path) -> Path:
        """Atomic download with retries; never promotes partial files."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        tmp = dest.with_suffix(dest.suffix + ".partial")
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                req = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(req, timeout=self.timeout_s) as resp:
                    data = resp.read()
                if not data:
                    raise EmptyResponseError(f"empty body: {url}")
                tmp.write_bytes(data)
                # basic ZIP magic check
                if data[:2] != b"PK":
                    tmp.unlink(missing_ok=True)
                    raise ProviderError(f"not a zip archive: {url}")
                tmp.replace(dest)
                time.sleep(self.pause_s)
                return dest
            except HTTPError as exc:
                last_err = exc
                if exc.code == 404:
                    tmp.unlink(missing_ok=True)
                    raise FileNotFoundError(url) from exc
                if exc.code == 429:
                    time.sleep(self.pause_s * attempt * 2)
                    continue
                time.sleep(self.pause_s * attempt)
            except (URLError, TimeoutError, OSError) as exc:
                last_err = exc
                time.sleep(self.pause_s * attempt)
        tmp.unlink(missing_ok=True)
        if isinstance(last_err, HTTPError) and last_err.code == 429:
            raise RateLimitError(str(last_err)) from last_err
        raise ProviderError(f"failed to download {url}: {last_err}") from last_err

    def _parse_zip(self, zip_path: Path, symbol: str) -> pd.DataFrame:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not names:
                raise ProviderError(f"no csv in {zip_path}")
            raw = zf.read(names[0])
        # detect header
        head = raw.split(b"\n", 1)[0].decode("utf-8", errors="ignore").lower()
        has_header = "open" in head and "time" in head
        buf = io.BytesIO(raw)
        if has_header:
            df = pd.read_csv(buf)
            # normalize column names
            cols = {c: c.strip().lower() for c in df.columns}
            df = df.rename(columns=cols)
            rename = {
                "open_time": "open_time",
                "open time": "open_time",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        else:
            df = pd.read_csv(buf, header=None, names=_KLINE_COLS)

        if "open_time" not in df.columns:
            raise ProviderError(f"missing open_time in {zip_path}")

        # open_time may be ms/us/ns int or already datetime string
        ot = df["open_time"]
        if pd.api.types.is_numeric_dtype(ot):
            # Magnitude heuristic (Binance Vision shifted from ms → µs around 2025):
            #   seconds  ~ 1e9
            #   ms       ~ 1e12
            #   µs       ~ 1e15
            #   ns       ~ 1e18
            sample = float(ot.iloc[0])
            if sample < 1e11:
                unit = "s"
            elif sample < 1e14:
                unit = "ms"
            elif sample < 1e17:
                unit = "us"
            else:
                unit = "ns"
            ts = pd.to_datetime(ot, unit=unit, utc=True)
        else:
            ts = pd.to_datetime(ot, utc=True)

        out = pd.DataFrame(
            {
                "timestamp": ts,
                "instrument": symbol.upper(),
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["volume"], errors="coerce"),
            }
        )
        if "count" in df.columns:
            out["trade_count"] = pd.to_numeric(df["count"], errors="coerce")
        return out.dropna(subset=["timestamp", "open", "high", "low", "close"])

    def download(self, request: ProviderRequest) -> ProviderResponse:
        symbol = str(request.original_symbol or request.instrument).upper().replace("-", "").replace("/", "")
        interval = str(request.frequency)
        if interval == "60m":
            interval = "1h"
        if interval not in SUPPORTED_INTERVALS:
            raise ProviderError(f"unsupported binance vision interval: {interval}")

        start = pd.Timestamp(request.start)
        end = pd.Timestamp(request.end)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")

        retrieval_ts = now_utc_iso()
        warnings: list[str] = []
        missing_months: list[str] = []
        frames: list[pd.DataFrame] = []

        months = _month_starts(start, end)
        for m in months:
            url = vision_monthly_url(symbol, interval, m.year, m.month)
            local = self.cache_dir / symbol / interval / f"{symbol}-{interval}-{m.year:04d}-{m.month:02d}.zip"
            try:
                path = self._download_zip(url, local)
                part = self._parse_zip(path, symbol)
                frames.append(part)
            except FileNotFoundError:
                missing_months.append(f"{m.year:04d}-{m.month:02d}")
                warnings.append(f"archive unavailable (404): {url}")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"month {m.year:04d}-{m.month:02d} failed: {exc}")
                # do not include partial corrupt month
                continue

        if not frames:
            raise EmptyResponseError(
                f"no Binance Vision months acquired for {symbol} {interval} "
                f"in [{start.date()}, {end.date()}]; missing={missing_months[:12]}"
            )

        # chunked concat
        frame = pd.concat(frames, ignore_index=True)
        frame = normalize_frame(frame)
        # filter requested range; do not fabricate gaps
        frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
        if frame.empty:
            raise EmptyResponseError("no rows remain after requested-range filter")

        # dedupe keep first (archive should be unique)
        before = len(frame)
        frame = frame.drop_duplicates(subset=["timestamp", "instrument"], keep="first")
        if len(frame) < before:
            warnings.append(f"dropped {before - len(frame)} duplicate timestamp rows")

        actual_start = str(frame["timestamp"].min())
        actual_end = str(frame["timestamp"].max())
        if missing_months:
            warnings.append(
                f"{len(missing_months)} month archive(s) unavailable; "
                f"actual range may be shorter than requested (not fabricated)."
            )

        return ProviderResponse(
            frame=frame,
            provider=self.provider_id,
            source="binance_vision_public_historical",
            retrieval_timestamp=retrieval_ts,
            requested_range=(str(start), str(end)),
            actual_range=(actual_start, actual_end),
            frequency=interval,
            timezone="UTC",
            original_timezone="UTC",
            exchange_timezone="UTC",
            adjustment_policy="unadjusted",
            original_symbol=symbol,
            normalized_symbol=symbol,
            currency="USDT",
            license_status="UNKNOWN",
            data_class="DEVELOPMENT/RESEARCH",
            rate_limit_info=self.rate_limits(),
            warnings=warnings,
            availability_timestamp_available=False,
            metadata={
                "market_type": "CRYPTO",
                "continuous_market": True,
                "session_model": "24x7",
                "data_tier": "DEVELOPMENT/RESEARCH",
                "missing_months": missing_months,
                "months_requested": len(months),
                "months_acquired": len(months) - len(missing_months),
                "archive_base": VISION_BASE,
                "api_key_required": False,
                "corporate_action_treatment": "n/a_crypto_spot",
                "not_institutional_grade": True,
            },
        )


__all__ = [
    "BinanceVisionHistoricalProvider",
    "VISION_BASE",
    "vision_monthly_url",
]
