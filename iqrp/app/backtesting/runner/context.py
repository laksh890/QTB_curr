"""Shared mutable simulation context for the event pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from iqrp.app.backtesting.accounting import (
    CapitalState,
    FillLog,
    OrderLog,
    PositionBook,
    SnapshotBook,
    TradeLedger,
)
from iqrp.app.backtesting.runner.adapters import (
    ExecutionSimulationAdapter,
    PortfolioConstructionAdapter,
)
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.strategy.base import Strategy


@dataclass
class PipelineContext:
    config: BacktestRunConfig
    strategy: Strategy
    capital: CapitalState
    positions: PositionBook
    orders: OrderLog = field(default_factory=OrderLog)
    fills: FillLog = field(default_factory=FillLog)
    trades: TradeLedger = field(default_factory=TradeLedger)
    snapshots: SnapshotBook = field(default_factory=SnapshotBook)
    portfolio_adapter: PortfolioConstructionAdapter = field(
        default_factory=PortfolioConstructionAdapter
    )
    execution_adapter: ExecutionSimulationAdapter = field(
        default_factory=ExecutionSimulationAdapter
    )
    universe: list[str] = field(default_factory=list)
    latest_prices: dict[str, float] = field(default_factory=dict)
    latest_bars: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_time: datetime | None = None
    target_weights: dict[str, float] = field(default_factory=dict)
    pending_orders: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    peak_equity: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    strategy_state: dict[str, Any] = field(default_factory=dict)
    risk_state: dict[str, Any] = field(default_factory=dict)
    model_state: dict[str, Any] = field(default_factory=dict)
    event_count: int = 0
    bar_count: int = 0
    pause_requested: bool = False
    cancel_requested: bool = False
    invalidated: bool = False
    invalidation_reason: str = ""
    last_turnover: float = 0.0
    last_costs: float = 0.0
    random_state: dict[str, Any] = field(default_factory=dict)
    engine: Any = None

    def mark_prices(self) -> None:
        self.positions.mark_all(self.latest_prices)
        mv = self.positions.total_market_value()
        unreal = self.positions.total_unrealized()
        self.capital.mark_unrealized(unreal, market_value=mv)

    def current_equity(self) -> float:
        self.mark_prices()
        return float(self.capital.equity)

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "current_time": None if self.current_time is None else self.current_time.isoformat(),
            "capital": self.capital.to_dict(),
            "positions": self.positions.to_dict(),
            "orders": self.orders.to_dict(),
            "fills": self.fills.to_dict(),
            "trades": self.trades.to_dict(),
            "snapshots": self.snapshots.to_dict(),
            "universe": list(self.universe),
            "latest_prices": dict(self.latest_prices),
            "target_weights": dict(self.target_weights),
            "equity_curve": list(self.equity_curve),
            "returns": list(self.returns),
            "timestamps": list(self.timestamps),
            "peak_equity": float(self.peak_equity),
            "diagnostics": dict(self.diagnostics),
            "strategy_state": dict(self.strategy_state),
            "risk_state": dict(self.risk_state),
            "model_state": dict(self.model_state),
            "event_count": int(self.event_count),
            "bar_count": int(self.bar_count),
            "random_state": dict(self.random_state),
            "pending_orders": list(self.pending_orders),
        }

    def load_checkpoint(self, data: dict[str, Any]) -> None:
        from datetime import datetime as dt

        cts = data.get("current_time")
        if cts:
            self.current_time = dt.fromisoformat(str(cts))
        self.capital = CapitalState.from_dict(data.get("capital") or {})
        self.positions = PositionBook.from_dict(data.get("positions") or {})
        self.orders = OrderLog.from_dict(data.get("orders") or {})
        self.fills = FillLog.from_dict(data.get("fills") or {})
        self.trades = TradeLedger.from_dict(data.get("trades") or {})
        self.snapshots = SnapshotBook.from_dict(data.get("snapshots") or {})
        self.universe = list(data.get("universe") or [])
        self.latest_prices = {str(k): float(v) for k, v in dict(data.get("latest_prices") or {}).items()}
        self.target_weights = {
            str(k): float(v) for k, v in dict(data.get("target_weights") or {}).items()
        }
        self.equity_curve = [float(x) for x in list(data.get("equity_curve") or [])]
        self.returns = [float(x) for x in list(data.get("returns") or [])]
        self.timestamps = [str(x) for x in list(data.get("timestamps") or [])]
        self.peak_equity = float(data.get("peak_equity") or self.capital.initial_capital)
        self.diagnostics = dict(data.get("diagnostics") or {})
        self.strategy_state = dict(data.get("strategy_state") or {})
        self.risk_state = dict(data.get("risk_state") or {})
        self.model_state = dict(data.get("model_state") or {})
        self.event_count = int(data.get("event_count") or 0)
        self.bar_count = int(data.get("bar_count") or 0)
        self.random_state = dict(data.get("random_state") or {})
        self.pending_orders = list(data.get("pending_orders") or [])


__all__ = ["PipelineContext"]
