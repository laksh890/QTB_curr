"""Operational backtest result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OperationalBacktestResult:
    backtest_id: str
    status: str
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    positions_log: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    capital: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    scenarios: dict[str, Any] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    initial_capital: float = 0.0
    seed: int = 42
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def pnl_changed(self) -> bool:
        if not self.equity_curve:
            return False
        start = float(self.initial_capital or self.equity_curve[0])
        end = float(self.equity_curve[-1])
        if abs(end - start) > 1e-6:
            return True
        # Also true if trading activity marked positions / costs even with flat PnL
        if self.fills or self.orders:
            return abs(end - start) > 1e-9 or bool(self.positions_log)
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "backtest_id": self.backtest_id,
            "status": self.status,
            "equity_curve": list(self.equity_curve),
            "returns": list(self.returns),
            "timestamps": list(self.timestamps),
            "orders": list(self.orders),
            "fills": list(self.fills),
            "trades": list(self.trades),
            "positions_log": list(self.positions_log),
            "snapshots": list(self.snapshots),
            "capital": dict(self.capital),
            "performance": dict(self.performance),
            "risk": dict(self.risk),
            "execution": dict(self.execution),
            "diagnostics": dict(self.diagnostics),
            "reports": dict(self.reports),
            "walk_forward": dict(self.walk_forward),
            "scenarios": dict(self.scenarios),
            "reconciliation": dict(self.reconciliation),
            "initial_capital": float(self.initial_capital),
            "seed": int(self.seed),
            "config": dict(self.config),
            "pnl_changed": bool(self.pnl_changed),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperationalBacktestResult":
        return cls(
            backtest_id=str(data.get("backtest_id", "")),
            status=str(data.get("status", "")),
            equity_curve=[float(x) for x in list(data.get("equity_curve") or [])],
            returns=[float(x) for x in list(data.get("returns") or [])],
            timestamps=[str(x) for x in list(data.get("timestamps") or [])],
            orders=list(data.get("orders") or []),
            fills=list(data.get("fills") or []),
            trades=list(data.get("trades") or []),
            positions_log=list(data.get("positions_log") or []),
            snapshots=list(data.get("snapshots") or []),
            capital=dict(data.get("capital") or {}),
            performance=dict(data.get("performance") or {}),
            risk=dict(data.get("risk") or {}),
            execution=dict(data.get("execution") or {}),
            diagnostics=dict(data.get("diagnostics") or {}),
            reports=dict(data.get("reports") or {}),
            walk_forward=dict(data.get("walk_forward") or {}),
            scenarios=dict(data.get("scenarios") or {}),
            reconciliation=dict(data.get("reconciliation") or {}),
            initial_capital=float(data.get("initial_capital", 0.0)),
            seed=int(data.get("seed", 42)),
            config=dict(data.get("config") or {}),
        )


__all__ = ["OperationalBacktestResult"]
