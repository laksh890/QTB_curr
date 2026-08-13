"""Trade ledger (round-trip entry/exit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TradeRecord:
    trade_id: str
    instrument: str
    side: str
    quantity: float
    entry_time: str
    entry_price: float
    exit_time: str | None = None
    exit_price: float | None = None
    realized_pnl: float = 0.0
    fees: float = 0.0
    status: str = "open"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "instrument": self.instrument,
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_time": self.entry_time,
            "entry_price": float(self.entry_price),
            "exit_time": self.exit_time,
            "exit_price": self.exit_price,
            "realized_pnl": float(self.realized_pnl),
            "fees": float(self.fees),
            "status": self.status,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeRecord":
        return cls(
            trade_id=str(data["trade_id"]),
            instrument=str(data.get("instrument", "")),
            side=str(data.get("side", "")),
            quantity=float(data.get("quantity", 0.0)),
            entry_time=str(data.get("entry_time", "")),
            entry_price=float(data.get("entry_price", 0.0)),
            exit_time=None if data.get("exit_time") is None else str(data["exit_time"]),
            exit_price=None if data.get("exit_price") is None else float(data["exit_price"]),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            fees=float(data.get("fees", 0.0)),
            status=str(data.get("status", "open")),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class TradeLedger:
    trades: list[TradeRecord] = field(default_factory=list)
    _open: dict[str, TradeRecord] = field(default_factory=dict, repr=False)

    def open_trade(
        self,
        *,
        trade_id: str,
        instrument: str,
        side: str,
        quantity: float,
        entry_time: str,
        entry_price: float,
        fees: float = 0.0,
    ) -> TradeRecord:
        rec = TradeRecord(
            trade_id=trade_id,
            instrument=instrument,
            side=side,
            quantity=float(quantity),
            entry_time=entry_time,
            entry_price=float(entry_price),
            fees=float(fees),
            status="open",
        )
        self.trades.append(rec)
        self._open[instrument] = rec
        return rec

    def close_trade(
        self,
        instrument: str,
        *,
        exit_time: str,
        exit_price: float,
        realized_pnl: float,
        fees: float = 0.0,
    ) -> TradeRecord | None:
        rec = self._open.pop(str(instrument), None)
        if rec is None:
            return None
        rec.exit_time = exit_time
        rec.exit_price = float(exit_price)
        rec.realized_pnl = float(realized_pnl)
        rec.fees += float(fees)
        rec.status = "closed"
        return rec

    def record_fill_as_trade(
        self,
        *,
        trade_id: str,
        instrument: str,
        side: str,
        quantity: float,
        timestamp: str,
        price: float,
        realized_pnl: float = 0.0,
        fees: float = 0.0,
    ) -> TradeRecord:
        """Append a ledger row for each fill (entry or exit style)."""
        status = "closed" if abs(float(realized_pnl)) > 1e-12 else "open"
        rec = TradeRecord(
            trade_id=trade_id,
            instrument=instrument,
            side=side,
            quantity=float(quantity),
            entry_time=timestamp,
            entry_price=float(price),
            exit_time=timestamp if status == "closed" else None,
            exit_price=float(price) if status == "closed" else None,
            realized_pnl=float(realized_pnl),
            fees=float(fees),
            status=status,
        )
        self.trades.append(rec)
        return rec

    def __len__(self) -> int:
        return len(self.trades)

    def to_list(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.trades]

    def to_dict(self) -> dict[str, Any]:
        return {"trades": self.to_list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeLedger":
        ledger = cls()
        for row in list(data.get("trades") or []):
            rec = TradeRecord.from_dict(row)
            ledger.trades.append(rec)
            if rec.status == "open":
                ledger._open[rec.instrument] = rec
        return ledger


__all__ = ["TradeLedger", "TradeRecord"]
