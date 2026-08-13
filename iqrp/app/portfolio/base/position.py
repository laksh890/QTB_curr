"""Portfolio position representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Position:
    """A single instrument holding with notional and weight metadata."""

    asset: str
    quantity: float = 0.0
    price: float = 0.0
    multiplier: float = 1.0
    lot_size: float = 1.0
    currency: str = "USD"
    notional: float | None = None
    weight: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.notional is None:
            self.notional = float(self.quantity) * float(self.price) * float(self.multiplier)

    def compute_notional(self) -> float:
        """Recompute notional from quantity, price, and multiplier."""
        self.notional = float(self.quantity) * float(self.price) * float(self.multiplier)
        return float(self.notional)

    def signed_notional(self) -> float:
        return float(self.notional if self.notional is not None else self.compute_notional())

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "multiplier": float(self.multiplier),
            "lot_size": float(self.lot_size),
            "currency": self.currency,
            "notional": float(self.notional if self.notional is not None else 0.0),
            "weight": float(self.weight),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Position:
        return cls(
            asset=str(data.get("asset", "")),
            quantity=float(data.get("quantity", 0.0)),
            price=float(data.get("price", 0.0)),
            multiplier=float(data.get("multiplier", 1.0)),
            lot_size=float(data.get("lot_size", 1.0)),
            currency=str(data.get("currency", "USD")),
            notional=float(data["notional"]) if data.get("notional") is not None else None,
            weight=float(data.get("weight", 0.0)),
            meta=dict(data.get("meta") or {}),
        )
