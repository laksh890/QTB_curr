"""Exchange adapter factory — configuration driven, no hard-coded selection."""

from __future__ import annotations

from collections.abc import Callable

from iqrp.app.config.settings import DataSettings, ExchangeEndpointSettings
from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.data.exchange.adapters.binance import BinanceExchange
from iqrp.app.data.exchange.adapters.bybit import BybitExchange
from iqrp.app.data.exchange.adapters.coinbase import CoinbaseExchange
from iqrp.app.data.exchange.base import BaseExchange

ExchangeBuilder = Callable[[ExchangeEndpointSettings], BaseExchange]

_REGISTRY: dict[str, ExchangeBuilder] = {
    "binance": BinanceExchange,
    "bybit": BybitExchange,
    "coinbase": CoinbaseExchange,
}


def register_exchange(name: str, builder: ExchangeBuilder) -> None:
    """Register or replace an exchange adapter builder."""
    _REGISTRY[name.lower()] = builder


def available_exchanges() -> tuple[str, ...]:
    """Return registered exchange names."""
    return tuple(sorted(_REGISTRY))


class ExchangeFactory:
    """Create :class:`BaseExchange` instances from :class:`DataSettings`."""

    def __init__(self, data_settings: DataSettings) -> None:
        self._settings = data_settings
        self._by_name = {item.name.lower(): item for item in data_settings.exchanges}

    def get_endpoint(self, name: str) -> ExchangeEndpointSettings:
        key = name.lower()
        if key not in self._by_name:
            raise ConfigurationError(
                f"Exchange '{name}' is not configured",
                code="EXCHANGE_NOT_CONFIGURED",
                details={"configured": sorted(self._by_name)},
            )
        return self._by_name[key]

    def create(self, name: str | None = None) -> BaseExchange:
        key = (name or self._settings.ingestion.default_exchange).lower()
        if key not in _REGISTRY:
            raise ConfigurationError(
                f"No adapter registered for exchange '{key}'",
                code="EXCHANGE_ADAPTER_MISSING",
                details={"available": list(available_exchanges())},
            )
        endpoint = self.get_endpoint(key)
        return _REGISTRY[key](endpoint)
