"""Daily / periodic portfolio snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class PortfolioSnapshot:
    timestamp: str
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    volatility: float = 0.0
    drawdown: float = 0.0
    var: float = 0.0
    cvar: float = 0.0
    turnover: float = 0.0
    costs: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "equity": float(self.equity),
            "cash": float(self.cash),
            "gross_exposure": float(self.gross_exposure),
            "net_exposure": float(self.net_exposure),
            "leverage": float(self.leverage),
            "volatility": float(self.volatility),
            "drawdown": float(self.drawdown),
            "var": float(self.var),
            "cvar": float(self.cvar),
            "turnover": float(self.turnover),
            "costs": float(self.costs),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "positions": {k: float(v) for k, v in self.positions.items()},
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PortfolioSnapshot":
        return cls(
            timestamp=str(data.get("timestamp", "")),
            equity=float(data.get("equity", 0.0)),
            cash=float(data.get("cash", 0.0)),
            gross_exposure=float(data.get("gross_exposure", 0.0)),
            net_exposure=float(data.get("net_exposure", 0.0)),
            leverage=float(data.get("leverage", 0.0)),
            volatility=float(data.get("volatility", 0.0)),
            drawdown=float(data.get("drawdown", 0.0)),
            var=float(data.get("var", 0.0)),
            cvar=float(data.get("cvar", 0.0)),
            turnover=float(data.get("turnover", 0.0)),
            costs=float(data.get("costs", 0.0)),
            realized_pnl=float(data.get("realized_pnl", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            positions={str(k): float(v) for k, v in dict(data.get("positions") or {}).items()},
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class SnapshotBook:
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)

    def add(self, snap: PortfolioSnapshot | Mapping[str, Any]) -> PortfolioSnapshot:
        rec = snap if isinstance(snap, PortfolioSnapshot) else PortfolioSnapshot.from_dict(snap)
        self.snapshots.append(rec)
        return rec

    def equity_curve(self) -> list[float]:
        return [float(s.equity) for s in self.snapshots]

    def __len__(self) -> int:
        return len(self.snapshots)

    def to_list(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.snapshots]

    def to_dict(self) -> dict[str, Any]:
        return {"snapshots": self.to_list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SnapshotBook":
        book = cls()
        for row in list(data.get("snapshots") or []):
            book.add(row)
        return book


__all__ = ["PortfolioSnapshot", "SnapshotBook"]
