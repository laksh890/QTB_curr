"""Provider registry (pluggable; framework not tied to a single vendor)."""

from __future__ import annotations

from typing import Callable

from iqrp.app.data.historical.provider import HistoricalDataProvider

_REGISTRY: dict[str, Callable[[], HistoricalDataProvider]] = {}


def register_provider(
    provider_id: str,
    factory: Callable[[], HistoricalDataProvider],
    *,
    overwrite: bool = False,
) -> None:
    key = str(provider_id).strip().lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"provider already registered: {key}")
    _REGISTRY[key] = factory


def get_provider(provider_id: str) -> HistoricalDataProvider:
    key = str(provider_id).strip().lower()
    if key not in _REGISTRY:
        # lazy defaults
        _ensure_defaults()
    if key not in _REGISTRY:
        raise KeyError(f"unknown provider: {provider_id!r}; known={list_providers()}")
    return _REGISTRY[key]()


def list_providers() -> list[str]:
    _ensure_defaults()
    return sorted(_REGISTRY.keys())


def _ensure_defaults() -> None:
    if "yahoo_finance" not in _REGISTRY and "yfinance" not in _REGISTRY:
        from iqrp.app.data.historical.yahoo_finance import YahooFinanceHistoricalProvider

        register_provider("yahoo_finance", YahooFinanceHistoricalProvider, overwrite=True)
        register_provider("yfinance", YahooFinanceHistoricalProvider, overwrite=True)
    if "binance" not in _REGISTRY and "binance_vision" not in _REGISTRY:
        from iqrp.app.data.historical.binance_vision import BinanceVisionHistoricalProvider

        register_provider("binance", BinanceVisionHistoricalProvider, overwrite=True)
        register_provider("binance_vision", BinanceVisionHistoricalProvider, overwrite=True)


__all__ = ["get_provider", "list_providers", "register_provider"]
