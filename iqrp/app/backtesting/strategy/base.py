"""Strategy ABC for the operational backtest runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class Strategy(ABC):
    """Point-in-time strategy interface wired into the event cascade.

    Implementations must never read data strictly after the event timestamp.
    Optional hooks may return a payload dict that the pipeline merges into
    subsequent events; ``None`` means no contribution.
    """

    strategy_id: str = ""
    strategy_version: str = "1.0.0"

    @abstractmethod
    def initialize(self, context: Any) -> None:
        """Called once after prepare, before the first market event."""

    def on_bar(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        """Optional bar-level hook (aggregated market snapshot)."""
        return None

    def on_market_data(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_features(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_signal(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_forecast(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_risk(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_portfolio(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_order(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_fill(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        return None

    def on_end(self, context: Any) -> Mapping[str, Any] | None:
        """Called after the simulation loop completes."""
        return None


__all__ = ["Strategy"]
