"""Portfolio representation for institutional construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np

from iqrp.app.portfolio.base.position import Position


class PortfolioType(str, Enum):
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"
    MARKET_NEUTRAL = "market_neutral"
    DOLLAR_NEUTRAL = "dollar_neutral"
    BETA_NEUTRAL = "beta_neutral"
    SECTOR_NEUTRAL = "sector_neutral"
    FACTOR_NEUTRAL = "factor_neutral"
    MULTI_STRATEGY = "multi_strategy"
    MULTI_ASSET = "multi_asset"


@dataclass(slots=True)
class Portfolio:
    """Target or current portfolio with names, weights, and positions."""

    names: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    positions: list[Position] = field(default_factory=list)
    cash: float = 0.0
    currency: str = "USD"
    portfolio_type: PortfolioType | str = PortfolioType.LONG_ONLY
    nav: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.portfolio_type, str):
            self.portfolio_type = PortfolioType(self.portfolio_type)
        if not self.names and self.positions:
            self.names = [p.asset for p in self.positions]
        if not self.weights and self.positions:
            self.weights = [float(p.weight) for p in self.positions]
        if len(self.weights) != len(self.names) and self.names:
            # Align weights to names when only partial weight data is present
            w = list(self.weights)
            if len(w) < len(self.names):
                w.extend([0.0] * (len(self.names) - len(w)))
            self.weights = w[: len(self.names)]

    @property
    def n_assets(self) -> int:
        return len(self.names)

    def weight_array(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64)

    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.weight_array())))

    def net_exposure(self) -> float:
        return float(np.sum(self.weight_array()))

    def long_exposure(self) -> float:
        w = self.weight_array()
        return float(np.sum(w[w > 0.0])) if w.size else 0.0

    def short_exposure(self) -> float:
        w = self.weight_array()
        return float(np.sum(np.abs(w[w < 0.0]))) if w.size else 0.0

    def leverage(self) -> float:
        return self.gross_exposure()

    def position_map(self) -> dict[str, Position]:
        return {p.asset: p for p in self.positions}

    def with_weights(
        self,
        weights: Iterable[float],
        *,
        names: Iterable[str] | None = None,
    ) -> Portfolio:
        name_list = list(names) if names is not None else list(self.names)
        w = [float(x) for x in weights]
        if len(name_list) != len(w):
            raise ValueError("names and weights must have equal length")
        pos_map = self.position_map()
        positions: list[Position] = []
        for name, weight in zip(name_list, w):
            if name in pos_map:
                p = pos_map[name]
                positions.append(
                    Position(
                        asset=p.asset,
                        quantity=p.quantity,
                        price=p.price,
                        multiplier=p.multiplier,
                        lot_size=p.lot_size,
                        currency=p.currency,
                        notional=p.notional,
                        weight=float(weight),
                        meta=dict(p.meta),
                    )
                )
            else:
                positions.append(Position(asset=name, weight=float(weight), currency=self.currency))
        return Portfolio(
            names=name_list,
            weights=w,
            positions=positions,
            cash=self.cash,
            currency=self.currency,
            portfolio_type=self.portfolio_type,
            nav=self.nav,
            meta=dict(self.meta),
        )

    def to_dict(self) -> dict[str, Any]:
        pt = self.portfolio_type.value if isinstance(self.portfolio_type, PortfolioType) else str(self.portfolio_type)
        return {
            "names": list(self.names),
            "weights": [float(w) for w in self.weights],
            "positions": [p.to_dict() for p in self.positions],
            "cash": float(self.cash),
            "currency": self.currency,
            "portfolio_type": pt,
            "nav": float(self.nav) if self.nav is not None else None,
            "meta": dict(self.meta),
            "gross_exposure": self.gross_exposure(),
            "net_exposure": self.net_exposure(),
            "long_exposure": self.long_exposure(),
            "short_exposure": self.short_exposure(),
            "leverage": self.leverage(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Portfolio:
        positions = [Position.from_dict(p) for p in (data.get("positions") or [])]
        pt = data.get("portfolio_type", PortfolioType.LONG_ONLY)
        return cls(
            names=list(data.get("names") or [p.asset for p in positions]),
            weights=[float(w) for w in (data.get("weights") or [p.weight for p in positions])],
            positions=positions,
            cash=float(data.get("cash", 0.0)),
            currency=str(data.get("currency", "USD")),
            portfolio_type=pt,
            nav=float(data["nav"]) if data.get("nav") is not None else None,
            meta=dict(data.get("meta") or {}),
        )

    @classmethod
    def cash_portfolio(cls, *, currency: str = "USD", cash: float = 1.0) -> Portfolio:
        return cls(
            names=[],
            weights=[],
            positions=[],
            cash=float(cash),
            currency=currency,
            portfolio_type=PortfolioType.LONG_ONLY,
            nav=float(cash),
            meta={"fallback": "cash"},
        )
