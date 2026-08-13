"""Fill audit log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class FillRecord:
    fill_id: str
    order_id: str
    timestamp: str
    instrument: str
    side: str
    quantity: float
    price: float
    fee: float = 0.0
    slippage: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "instrument": self.instrument,
            "side": self.side,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "fee": float(self.fee),
            "slippage": float(self.slippage),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FillRecord":
        return cls(
            fill_id=str(data.get("fill_id", data.get("event_id", ""))),
            order_id=str(data.get("order_id", "")),
            timestamp=str(data.get("timestamp", "")),
            instrument=str(data.get("instrument", data.get("symbol", ""))),
            side=str(data.get("side", "")),
            quantity=float(data.get("quantity", data.get("qty", 0.0))),
            price=float(data.get("price", data.get("fill_price", 0.0))),
            fee=float(data.get("fee", 0.0)),
            slippage=float(data.get("slippage", 0.0)),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class FillLog:
    fills: list[FillRecord] = field(default_factory=list)

    def add(self, record: FillRecord | Mapping[str, Any]) -> FillRecord:
        rec = record if isinstance(record, FillRecord) else FillRecord.from_dict(record)
        self.fills.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self.fills)

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.fills]

    def to_dict(self) -> dict[str, Any]:
        return {"fills": self.to_list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FillLog":
        log = cls()
        rows = data.get("fills") if isinstance(data, Mapping) else data
        for row in list(rows or []):
            log.add(row)
        return log


__all__ = ["FillLog", "FillRecord"]
