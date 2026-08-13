"""Historical slippage calibration and lookup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(slots=True)
class HistoricalSlippageRecord:
    participation: float
    slippage_bps: float
    spread_bps: float = 0.0
    volatility: float = 0.0
    side: str = "buy"
    metadata: dict[str, Any] = field(default_factory=dict)


class HistoricalSlippageModel:
    """Nonparametric / binned historical slippage estimator."""

    def __init__(
        self,
        records: Sequence[HistoricalSlippageRecord | dict[str, Any]] | None = None,
        *,
        default_bps: float = 5.0,
    ) -> None:
        self.default_bps = float(default_bps)
        self._participation: list[float] = []
        self._slippage_bps: list[float] = []
        if records:
            for r in records:
                self.add(r)

    def add(self, record: HistoricalSlippageRecord | dict[str, Any]) -> None:
        if isinstance(record, HistoricalSlippageRecord):
            self._participation.append(float(record.participation))
            self._slippage_bps.append(float(record.slippage_bps))
        else:
            self._participation.append(float(record.get("participation", 0.0)))
            self._slippage_bps.append(float(record.get("slippage_bps", self.default_bps)))

    def estimate_bps(
        self,
        *,
        quantity: float,
        adv: float,
        k_neighbors: int = 8,
    ) -> dict[str, float]:
        """k-NN style estimate in participation space."""
        part = abs(float(quantity)) / max(float(adv), 1e-12)
        if not self._participation:
            return {
                "expected_slippage_bps": self.default_bps,
                "participation": float(part),
                "n_obs": 0.0,
            }
        p = np.asarray(self._participation, dtype=np.float64)
        s = np.asarray(self._slippage_bps, dtype=np.float64)
        dist = np.abs(p - part)
        k = min(max(int(k_neighbors), 1), p.size)
        idx = np.argpartition(dist, k - 1)[:k]
        w = 1.0 / np.maximum(dist[idx], 1e-8)
        w = w / float(np.sum(w))
        est = float(np.dot(w, s[idx]))
        return {
            "expected_slippage_bps": est,
            "participation": float(part),
            "n_obs": float(p.size),
            "k": float(k),
        }

    def calibrate_linear(self) -> dict[str, float]:
        """Fit slippage_bps ≈ a + b * sqrt(participation)."""
        if len(self._participation) < 2:
            return {"intercept": self.default_bps, "slope": 0.0, "r2": 0.0}
        x = np.sqrt(np.maximum(np.asarray(self._participation, dtype=np.float64), 0.0))
        y = np.asarray(self._slippage_bps, dtype=np.float64)
        A = np.column_stack([np.ones_like(x), x])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            "intercept": float(coef[0]),
            "slope": float(coef[1]),
            "r2": float(r2),
            "n_obs": float(len(y)),
        }


def historical_slippage_bps(
    quantity: float,
    adv: float,
    *,
    records: Sequence[HistoricalSlippageRecord | dict[str, Any]] | None = None,
    default_bps: float = 5.0,
) -> float:
    model = HistoricalSlippageModel(records, default_bps=default_bps)
    return float(model.estimate_bps(quantity=quantity, adv=adv)["expected_slippage_bps"])


__all__ = [
    "HistoricalSlippageModel",
    "HistoricalSlippageRecord",
    "historical_slippage_bps",
]
