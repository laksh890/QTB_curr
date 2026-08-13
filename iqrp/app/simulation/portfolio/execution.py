"""Simulated portfolio execution against synthetic markets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from iqrp.app.simulation.liquidity.slippage import SlippageModel


@dataclass
class Fill:
    timestamp: Any
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    temporary_impact: float
    permanent_impact: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    fills: list[Fill]
    total_notional: float
    average_slippage_bps: float
    frame: pl.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_fills": len(self.fills),
            "total_notional": self.total_notional,
            "average_slippage_bps": self.average_slippage_bps,
        }


class SimulatedExecutionEngine:
    """Execute child orders on simulated mid/spread/volume paths."""

    def __init__(
        self,
        *,
        impact: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.slippage = SlippageModel(impact=impact, rng=rng)
        self.rng = rng or np.random.default_rng()

    def execute_twap(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        mids: np.ndarray,
        spreads: np.ndarray,
        volumes: np.ndarray,
        volatility: np.ndarray,
        timestamps: list[Any],
        n_slices: int = 10,
    ) -> ExecutionReport:
        mids_a = np.asarray(mids, dtype=np.float64)
        n = len(mids_a)
        slices = max(1, min(n_slices, n))
        idx = np.linspace(0, n - 1, slices).astype(int)
        qty_each = quantity / slices
        fills: list[Fill] = []
        slip_bps: list[float] = []
        for i in idx:
            vol = float(volatility[min(i, len(volatility) - 1)])
            spread = float(spreads[min(i, len(spreads) - 1)])
            adv = float(volumes[min(i, len(volumes) - 1)]) * 24.0
            mid = float(mids_a[i])
            result = self.slippage.execution_price(
                mid, side, qty_each, adv=max(adv, 1e-8), volatility=vol, spread=spread
            )
            fills.append(
                Fill(
                    timestamp=timestamps[i] if i < len(timestamps) else None,
                    symbol=symbol,
                    side=side,
                    quantity=qty_each,
                    price=result["price"],
                    temporary_impact=result["temporary_impact"],
                    permanent_impact=result["permanent_impact"],
                )
            )
            slip_bps.append(10_000.0 * (result["price"] - mid) / mid * (1 if side == "buy" else -1))
        frame = pl.DataFrame(
            {
                "timestamp": [f.timestamp for f in fills],
                "symbol": [f.symbol for f in fills],
                "side": [f.side for f in fills],
                "quantity": [f.quantity for f in fills],
                "price": [f.price for f in fills],
                "temporary_impact": [f.temporary_impact for f in fills],
                "permanent_impact": [f.permanent_impact for f in fills],
            }
        )
        notional = float(sum(f.price * f.quantity for f in fills))
        return ExecutionReport(
            fills=fills,
            total_notional=notional,
            average_slippage_bps=float(np.mean(slip_bps)) if slip_bps else 0.0,
            frame=frame,
        )
