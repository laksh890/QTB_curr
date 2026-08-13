"""Shared execution enums and kill-switch registry.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Execution never overrides hard risk limits.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Kill switches are fail-safe and auditable.
- No future information may enter execution decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Side(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"

    @property
    def is_buy(self) -> bool:
        return self in (Side.BUY, Side.COVER)

    @property
    def signed_direction(self) -> int:
        return 1 if self.is_buy else -1

    @classmethod
    def parse(cls, value: object) -> Side:
        if isinstance(value, cls):
            return value
        text = str(value).strip().upper()
        if text in {"B", "BUY", "LONG"}:
            return cls.BUY
        if text in {"S", "SELL"}:
            return cls.SELL
        if text == "SHORT":
            return cls.SHORT
        if text in {"COVER", "BUY_TO_COVER"}:
            return cls.COVER
        raise ValueError(f"unsupported side: {value!r}")


class Urgency(str, Enum):
    """Execution urgency. Influences aggressiveness; never overrides hard risk."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class OrderType(str, Enum):
    """Supported order-type abstractions. Availability is venue-dependent."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    IOC = "IOC"
    FOK = "FOK"
    GTC = "GTC"
    GTT = "GTT"
    GTD = "GTD"
    POST_ONLY = "POST_ONLY"
    REDUCE_ONLY = "REDUCE_ONLY"
    ICEBERG = "ICEBERG"
    LOC = "LOC"  # limit-on-close
    MOC = "MOC"  # market-on-close
    MOO = "MOO"  # market-on-open
    PEGGED = "PEGGED"

    @classmethod
    def parse(cls, value: object) -> OrderType:
        if isinstance(value, cls):
            return value
        text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        aliases = {
            "LIMIT_ON_CLOSE": "LOC",
            "MARKET_ON_CLOSE": "MOC",
            "MARKET_ON_OPEN": "MOO",
            "STOPLIMIT": "STOP_LIMIT",
            "POSTONLY": "POST_ONLY",
            "REDUCEONLY": "REDUCE_ONLY",
        }
        text = aliases.get(text, text)
        try:
            return cls[text]
        except KeyError as exc:
            raise ValueError(f"unsupported order type: {value!r}") from exc


class TimeInForce(str, Enum):
    """Time-in-force policy."""

    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTT = "GTT"
    GTD = "GTD"
    OPG = "OPG"
    CLS = "CLS"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KillSwitch:
    """Fail-safe kill-switch registry (global / account / venue / strategy).

    When engaged, new order submission and routing must be blocked.
    """

    global_halt: bool = False
    reason: str = ""
    accounts: set[str] = field(default_factory=set)
    venues: set[str] = field(default_factory=set)
    strategies: set[str] = field(default_factory=set)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def engage_global(self, reason: str = "global kill-switch") -> None:
        self.global_halt = True
        self.reason = str(reason)
        self._audit("engage_global", reason=self.reason)

    # Aliases used by OrderManager docs / callers
    def halt_global(self, reason: str = "global kill-switch") -> None:
        self.engage_global(reason)

    def clear_global(self) -> None:
        self.global_halt = False
        self.reason = ""
        self._audit("clear_global")

    def engage_account(self, account_id: str, reason: str = "account kill-switch") -> None:
        self.accounts.add(str(account_id))
        self._audit("engage_account", account_id=str(account_id), reason=reason)

    def halt_account(self, account_id: str, reason: str = "") -> None:
        self.engage_account(account_id, reason or "account kill-switch")

    def clear_account(self, account_id: str) -> None:
        self.accounts.discard(str(account_id))
        self._audit("clear_account", account_id=str(account_id))

    def engage_venue(self, venue: str, reason: str = "venue kill-switch") -> None:
        self.venues.add(str(venue))
        self._audit("engage_venue", venue=str(venue), reason=reason)

    def halt_venue(self, venue: str, reason: str = "") -> None:
        self.engage_venue(venue, reason or "venue kill-switch")

    def clear_venue(self, venue: str) -> None:
        self.venues.discard(str(venue))
        self._audit("clear_venue", venue=str(venue))

    def engage_strategy(self, strategy_id: str, reason: str = "strategy kill-switch") -> None:
        self.strategies.add(str(strategy_id))
        self._audit("engage_strategy", strategy_id=str(strategy_id), reason=reason)

    def halt_strategy(self, strategy_id: str, reason: str = "") -> None:
        self.engage_strategy(strategy_id, reason or "strategy kill-switch")

    def clear_strategy(self, strategy_id: str) -> None:
        self.strategies.discard(str(strategy_id))
        self._audit("clear_strategy", strategy_id=str(strategy_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_halt": self.global_halt,
            "accounts": sorted(self.accounts),
            "venues": sorted(self.venues),
            "strategies": sorted(self.strategies),
            "reason": self.reason,
            "audit": list(self.audit),
        }

    def is_blocked(
        self,
        *,
        account_id: str | None = None,
        venue: str | None = None,
        strategy_id: str | None = None,
    ) -> tuple[bool, str]:
        """Return ``(blocked, reason)`` for the given scopes."""
        if self.global_halt:
            return True, self.reason or "global kill-switch active"
        if account_id is not None and str(account_id) in self.accounts:
            return True, f"account kill-switch active: {account_id}"
        if venue is not None and str(venue) in self.venues:
            return True, f"venue kill-switch active: {venue}"
        if strategy_id is not None and str(strategy_id) in self.strategies:
            return True, f"strategy kill-switch active: {strategy_id}"
        return False, ""

    def _audit(self, action: str, **details: Any) -> None:
        self.audit.append(
            {
                "action": action,
                "timestamp": _utc_now(),
                "details": dict(details),
                "global_halt": self.global_halt,
            }
        )


__all__ = [
    "Side",
    "Urgency",
    "OrderType",
    "TimeInForce",
    "KillSwitch",
]
