"""Capital / cash / equity accounting for operational backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CapitalState:
    """Mutable capital ledger."""

    initial_capital: float
    currency: str = "USD"
    cash: float = 0.0
    margin_used: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    financing_paid: float = 0.0
    position_market_value: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial_capital = float(self.initial_capital)
        if self.cash == 0.0 and "cash_initialized" not in self.meta:
            self.cash = float(self.initial_capital)
            self.meta["cash_initialized"] = True

    @property
    def equity(self) -> float:
        # Cash is reduced by purchase notional; equity = cash + position MV.
        return float(self.cash) + float(self.position_market_value)

    @property
    def available_cash(self) -> float:
        return float(self.cash) - float(self.margin_used)

    @property
    def free_cash(self) -> float:
        return max(float(self.cash) - float(self.margin_used), 0.0)

    @property
    def margin(self) -> float:
        return float(self.margin_used)

    def apply_cash_delta(self, delta: float, *, reason: str = "") -> None:
        self.cash = float(self.cash) + float(delta)
        if reason:
            self.meta.setdefault("cash_events", []).append(
                {"delta": float(delta), "reason": reason, "cash": float(self.cash)}
            )

    def record_fee(self, fee: float) -> None:
        fee = abs(float(fee))
        self.fees_paid += fee
        self.cash -= fee

    def record_financing(self, cost: float) -> None:
        cost = float(cost)
        self.financing_paid += abs(cost)
        self.cash -= abs(cost)

    def mark_unrealized(self, unrealized: float, *, market_value: float | None = None) -> None:
        self.unrealized_pnl = float(unrealized)
        if market_value is not None:
            self.position_market_value = float(market_value)

    def realize(self, pnl: float, *, settle_into_cash: bool = False) -> None:
        """Accumulate realized PnL. Cash settlement is optional (fills usually settle)."""
        pnl = float(pnl)
        self.realized_pnl += pnl
        if settle_into_cash:
            self.cash += pnl

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": float(self.initial_capital),
            "currency": self.currency,
            "cash": float(self.cash),
            "available_cash": float(self.available_cash),
            "free_cash": float(self.free_cash),
            "margin": float(self.margin),
            "margin_used": float(self.margin_used),
            "equity": float(self.equity),
            "position_market_value": float(self.position_market_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "fees_paid": float(self.fees_paid),
            "financing_paid": float(self.financing_paid),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapitalState":
        obj = cls(
            initial_capital=float(data.get("initial_capital", 0.0)),
            currency=str(data.get("currency", "USD")),
            cash=float(data.get("cash", data.get("initial_capital", 0.0))),
            margin_used=float(data.get("margin_used", data.get("margin", 0.0))),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            fees_paid=float(data.get("fees_paid", 0.0)),
            financing_paid=float(data.get("financing_paid", 0.0)),
            position_market_value=float(data.get("position_market_value", 0.0)),
            meta={"cash_initialized": True},
        )
        return obj


__all__ = ["CapitalState"]
