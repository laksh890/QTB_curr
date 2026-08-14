"""Order audit log."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class OrderRecord:
    order_id: str
    timestamp: str
    instrument: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    status: str = "created"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "instrument": self.instrument,
            "side": self.side,
            "quantity": float(self.quantity),
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "status": self.status,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrderRecord:
        return cls(
            order_id=str(data["order_id"]),
            timestamp=str(data.get("timestamp", "")),
            instrument=str(data.get("instrument", "")),
            side=str(data.get("side", "")),
            quantity=float(data.get("quantity", 0.0)),
            order_type=str(data.get("order_type", "market")),
            limit_price=None if data.get("limit_price") is None else float(data["limit_price"]),
            status=str(data.get("status", "created")),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class OrderLog:
    orders: list[OrderRecord] = field(default_factory=list)

    def add(self, record: OrderRecord | Mapping[str, Any]) -> OrderRecord:
        rec = record if isinstance(record, OrderRecord) else OrderRecord.from_dict(record)
        self.orders.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self.orders)

    def to_list(self) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.orders]

    def to_dict(self) -> dict[str, Any]:
        return {"orders": self.to_list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrderLog:
        log = cls()
        if isinstance(data, list):
            rows = data
        else:
            rows = list(data.get("orders") or [])
        for row in rows:
            log.add(row)
        return log

    @staticmethod
    def ts_str(ts: datetime | str) -> str:
        if isinstance(ts, datetime):
            return ts.isoformat()
        return str(ts)


__all__ = ["OrderLog", "OrderRecord"]
