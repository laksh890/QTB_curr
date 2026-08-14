"""Convert target weights to tradeable quantities / positions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from iqrp.app.portfolio.base.position import Position
from iqrp.app.portfolio.construction.target_weights import TargetWeights


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class TargetPositions:
    """Bundle of Position objects sized from target weights and capital."""

    positions: list[Position] = field(default_factory=list)
    capital: float = 0.0
    currency: str = "USD"
    timestamp: str = field(default_factory=_utc_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.positions)

    def __len__(self) -> int:
        return len(self.positions)

    def names(self) -> list[str]:
        return [p.asset for p in self.positions]

    def quantities(self) -> list[float]:
        return [float(p.quantity) for p in self.positions]

    def weights(self) -> list[float]:
        return [float(p.weight) for p in self.positions]

    def gross_notional(self) -> float:
        return float(sum(abs(float(p.notional or 0.0)) for p in self.positions))

    def net_notional(self) -> float:
        return float(sum(float(p.notional or 0.0) for p in self.positions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": [p.to_dict() for p in self.positions],
            "capital": float(self.capital),
            "currency": self.currency,
            "timestamp": self.timestamp,
            "meta": dict(self.meta),
            "gross_notional": self.gross_notional(),
            "net_notional": self.net_notional(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetPositions:
        return cls(
            positions=[Position.from_dict(p) for p in (data.get("positions") or [])],
            capital=float(data.get("capital", 0.0)),
            currency=str(data.get("currency", "USD")),
            timestamp=str(data.get("timestamp", _utc_now())),
            meta=dict(data.get("meta") or {}),
        )


def _align(arr: Any | None, n: int, default: float) -> np.ndarray:
    if arr is None:
        return np.full(n, float(default), dtype=np.float64)
    v = np.asarray(arr, dtype=np.float64).reshape(-1)
    out = np.full(n, float(default), dtype=np.float64)
    m = min(n, v.size)
    out[:m] = v[:m]
    return out


def weights_to_positions(
    weights: Sequence[float] | np.ndarray | TargetWeights | dict[str, float],
    *,
    capital: float,
    prices: Sequence[float] | np.ndarray,
    names: Sequence[str] | None = None,
    multipliers: Sequence[float] | np.ndarray | None = None,
    lot_sizes: Sequence[float] | np.ndarray | None = None,
    fx_rates: Sequence[float] | np.ndarray | None = None,
    currencies: Sequence[str] | None = None,
    min_order: float | Sequence[float] | None = None,
    max_order: float | Sequence[float] | None = None,
    round_lots: bool = True,
    currency: str = "USD",
) -> TargetPositions:
    """Convert portfolio weights to quantities given capital and prices.

    quantity ≈ (weight * capital) / (price * multiplier * fx)
    then rounded to lot_size; clipped by min/max order (absolute quantity).
    """
    if isinstance(weights, TargetWeights):
        name_list = list(weights.names)
        w = weights.as_array()
    elif isinstance(weights, dict):
        name_list = list(names) if names is not None else list(weights.keys())
        w = np.asarray([float(weights.get(nm, 0.0)) for nm in name_list], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        name_list = list(names) if names is not None else [f"a{i}" for i in range(w.size)]

    n = int(w.size)
    if len(name_list) != n:
        name_list = [f"a{i}" for i in range(n)]

    px = _align(prices, n, 1.0)
    mult = _align(multipliers, n, 1.0)
    lots = _align(lot_sizes, n, 1.0)
    fx = _align(fx_rates, n, 1.0)
    lots = np.maximum(lots, 1e-12)
    px = np.where(np.abs(px) < 1e-12, 1e-12, px)
    mult = np.where(np.abs(mult) < 1e-12, 1.0, mult)
    fx = np.where(np.abs(fx) < 1e-12, 1.0, fx)

    if min_order is None:
        min_o = np.zeros(n, dtype=np.float64)
    elif isinstance(min_order, (int, float)):
        min_o = np.full(n, float(min_order), dtype=np.float64)
    else:
        min_o = _align(min_order, n, 0.0)

    if max_order is None:
        max_o = np.full(n, np.inf, dtype=np.float64)
    elif isinstance(max_order, (int, float)):
        max_o = np.full(n, float(max_order), dtype=np.float64)
    else:
        max_o = _align(max_order, n, np.inf)

    cap = float(capital)
    notionals = w * cap
    denom = px * mult * fx
    raw_qty = notionals / denom

    positions: list[Position] = []
    cur_list = list(currencies) if currencies is not None else [currency] * n
    if len(cur_list) < n:
        cur_list = cur_list + [currency] * (n - len(cur_list))

    for i in range(n):
        q = float(raw_qty[i])
        lot = float(lots[i])
        if round_lots and lot > 0:
            q = float(np.sign(q) * np.floor(abs(q) / lot + 1e-12) * lot)
        # min order: drop trades smaller than min (absolute)
        if abs(q) + 1e-15 < float(min_o[i]) and abs(q) > 0:
            q = 0.0
        if np.isfinite(max_o[i]):
            q = float(np.clip(q, -float(max_o[i]), float(max_o[i])))
        price = float(px[i])
        multiplier = float(mult[i])
        notional = q * price * multiplier * float(fx[i])
        weight = float(notional / cap) if abs(cap) > 1e-12 else float(w[i])
        positions.append(
            Position(
                asset=name_list[i],
                quantity=q,
                price=price,
                multiplier=multiplier,
                lot_size=lot,
                currency=str(cur_list[i]),
                notional=float(notional),
                weight=weight,
                meta={"fx": float(fx[i]), "target_weight": float(w[i])},
            )
        )

    return TargetPositions(
        positions=positions,
        capital=cap,
        currency=currency,
        meta={
            "round_lots": bool(round_lots),
            "n_assets": n,
        },
    )


def target_positions(
    weights: Sequence[float] | np.ndarray | TargetWeights | dict[str, float],
    **kwargs: Any,
) -> TargetPositions:
    """Alias for :func:`weights_to_positions`."""
    return weights_to_positions(weights, **kwargs)
