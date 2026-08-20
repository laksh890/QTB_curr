"""Yahoo Finance / yfinance development historical provider.

DEVELOPMENT DATA only — not institutional-grade.
License status: UNKNOWN (do not fabricate licensing claims).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

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
from iqrp.app.data.historical.timestamps import ensure_aware_utc

# Normalized research symbols → Yahoo tickers (extensible; not hard-coded into generic layer)
DEFAULT_SYMBOL_MAP: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "^NSEI": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "^NSEBANK": "^NSEBANK",
}

YF_INTERVAL: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "60m": "60m",
    "1D": "1d",
    "1d": "1d",
}

# Documented Yahoo limitations (approximate; enforced server-side)
YF_MAX_WINDOW: dict[str, timedelta] = {
    "1m": timedelta(days=8),
    "5m": timedelta(days=60),
    "15m": timedelta(days=60),
    "30m": timedelta(days=60),
    "1h": timedelta(days=730),
    "60m": timedelta(days=730),
    "1d": timedelta(days=365 * 30),
    "1D": timedelta(days=365 * 30),
}


class YahooFinanceHistoricalProvider(HistoricalDataProvider):
    provider_id = "yahoo_finance"

    def __init__(
        self,
        *,
        symbol_map: dict[str, str] | None = None,
        exchange_timezone: str = "Asia/Kolkata",
    ) -> None:
        self.symbol_map = {**(symbol_map or DEFAULT_SYMBOL_MAP)}
        self.exchange_timezone = exchange_timezone

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.provider_id,
            supported_frequencies=tuple(YF_INTERVAL.keys()),
            data_class=data_class_label(institutional=False),
            rate_limits=self.rate_limits(),
            notes=(
                "Free/development Yahoo Finance via yfinance.",
                "1m history typically limited to ~7–8 days per request.",
                "Intraday ranges generally limited (e.g. 5m ≈ 60 days).",
                "Not institutional-grade. License status UNKNOWN.",
                "Availability timestamps are NOT provided by this source.",
            ),
            license_status="UNKNOWN",
        )

    def list_instruments(self) -> list[str]:
        return sorted({k for k in self.symbol_map if not k.startswith("^")})

    def supported_frequencies(self) -> tuple[str, ...]:
        return tuple(YF_INTERVAL.keys())

    def rate_limits(self) -> dict[str, Any]:
        return {
            "note": "Yahoo unofficial rate limits; backoff on HTTP 429.",
            "1m_max_window_days": 8,
            "5m_max_window_days": 60,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider_id,
            "backend": "yfinance",
            "data_class": data_class_label(institutional=False),
            "license_status": "UNKNOWN",
            "symbol_map": dict(self.symbol_map),
        }

    def available_history(self, instrument: str, frequency: str) -> dict[str, Any]:
        freq = str(frequency)
        window = YF_MAX_WINDOW.get(freq, timedelta(days=8))
        end = datetime.now(UTC)
        start = end - window
        return {
            "instrument": instrument,
            "frequency": freq,
            "approximate_max_window": str(window),
            "suggested_start": start.date().isoformat(),
            "suggested_end": end.date().isoformat(),
            "limitation": "Yahoo rolling window; older intraday bars unavailable via this free API.",
        }

    def resolve_symbol(self, instrument: str) -> str:
        key = str(instrument).strip()
        if key in self.symbol_map:
            return self.symbol_map[key]
        # pass-through Yahoo symbols
        return key

    def download(self, request: ProviderRequest) -> ProviderResponse:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("yfinance is required for YahooFinanceHistoricalProvider") from exc

        freq = str(request.frequency)
        if freq not in YF_INTERVAL:
            raise ProviderError(f"unsupported frequency for yahoo_finance: {freq}")

        original = request.original_symbol or self.resolve_symbol(request.instrument)
        normalized = str(request.instrument).strip()
        interval = YF_INTERVAL[freq]
        start = pd.Timestamp(request.start)
        end = pd.Timestamp(request.end)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")

        auto_adjust = str(request.adjustment_policy).lower() in {"adjusted", "auto_adjust", "true"}
        retrieval_ts = now_utc_iso()
        warnings: list[str] = []

        try:
            raw = yf.download(
                original,
                start=start.tz_convert("UTC").strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).tz_convert("UTC").strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg:
                raise RateLimitError(str(exc)) from exc
            raise ProviderError(str(exc)) from exc

        if raw is None or raw.empty:
            # try period= for short 1m windows
            if freq == "1m":
                raw = yf.download(
                    original,
                    period="8d",
                    interval="1m",
                    auto_adjust=auto_adjust,
                    progress=False,
                    threads=False,
                )
                warnings.append("fell back to period=8d for 1m due to empty start/end response")
            if raw is None or raw.empty:
                raise EmptyResponseError(
                    f"yahoo_finance returned empty data for {original} freq={freq}"
                )

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()
        # Datetime column name varies
        ts_col = None
        for c in df.columns:
            if str(c).lower() in {"datetime", "date", "index"}:
                ts_col = c
                break
        if ts_col is None:
            ts_col = df.columns[0]

        rename = {
            ts_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["instrument"] = normalized

        # Preserve original tz info before UTC conversion
        ts_record: dict[str, Any] = {}
        raw_ts = df["timestamp"]
        # yfinance often returns tz-aware Asia/Kolkata for NSE
        sample = pd.to_datetime(raw_ts.iloc[0]) if len(df) else None
        original_tz = "UNKNOWN"
        if sample is not None and getattr(sample, "tzinfo", None) is not None:
            original_tz = str(sample.tzinfo)
            df["timestamp"] = ensure_aware_utc(df["timestamp"], record=ts_record)
        else:
            # require explicit exchange timezone for naive
            df["timestamp"] = ensure_aware_utc(
                df["timestamp"],
                assume_timezone=request.exchange_timezone or self.exchange_timezone,
                record=ts_record,
            )
            original_tz = "naive→" + (request.exchange_timezone or self.exchange_timezone)
            warnings.append(ts_record.get("timestamp_conversion", "naive localized"))

        # Filter to requested range in UTC
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
        if df.empty:
            raise EmptyResponseError("no rows remain after requested-range filter")

        keep = ["timestamp", "instrument", "open", "high", "low", "close", "volume"]
        if "adj_close" in df.columns and auto_adjust is False:
            keep.append("adj_close")
        df = df[[c for c in keep if c in df.columns]]
        frame = normalize_frame(df)

        actual_start = str(frame["timestamp"].min()) if len(frame) else None
        actual_end = str(frame["timestamp"].max()) if len(frame) else None

        adj_status = "adjusted" if auto_adjust else "unadjusted"
        if not auto_adjust:
            warnings.append(
                "NIFTY index via Yahoo: adjustment semantics are source-defined; "
                "dataset marked unadjusted (auto_adjust=False)."
            )

        return ProviderResponse(
            frame=frame,
            provider=self.provider_id,
            source="yahoo_finance/yfinance",
            retrieval_timestamp=retrieval_ts,
            requested_range=(str(start), str(end)),
            actual_range=(actual_start, actual_end),
            frequency=freq if freq != "60m" else "1h",
            timezone="UTC",
            original_timezone=original_tz,
            exchange_timezone=request.exchange_timezone or self.exchange_timezone,
            adjustment_policy=adj_status,
            original_symbol=original,
            normalized_symbol=normalized,
            currency="INR",
            license_status="UNKNOWN",
            data_class=data_class_label(institutional=False),
            rate_limit_info=self.rate_limits(),
            warnings=warnings,
            availability_timestamp_available=False,
            metadata={
                "yfinance_interval": interval,
                "timestamp_conversion": ts_record,
                "corporate_action_treatment": "UNKNOWN" if not auto_adjust else "yahoo_auto_adjust",
                "not_institutional_grade": True,
            },
        )


__all__ = ["DEFAULT_SYMBOL_MAP", "YahooFinanceHistoricalProvider"]
