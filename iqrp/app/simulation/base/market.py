"""Simulated market objects and ground-truth containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl


@dataclass
class GroundTruth:
    """Oracle labels for every simulation - used to score future models."""

    regime_ids: np.ndarray
    regime_names: tuple[str, ...]
    volatility: np.ndarray
    drift: np.ndarray
    trend: np.ndarray  # sign/strength of local trend
    transition_matrix: np.ndarray
    event_mask: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self, timestamp_column: str = "open_time") -> pl.DataFrame:
        n = len(self.regime_ids)
        data: dict[str, Any] = {
            "true_regime": self.regime_ids.tolist(),
            "true_regime_name": [
                self.regime_names[int(i)] if int(i) < len(self.regime_names) else str(i)
                for i in self.regime_ids
            ],
            "true_volatility": (
                np.asarray(self.volatility).reshape(n, -1)[:, 0].tolist()
                if np.asarray(self.volatility).ndim > 1
                else np.asarray(self.volatility).tolist()
            ),
            "true_drift": (
                np.asarray(self.drift).reshape(n, -1)[:, 0].tolist()
                if np.asarray(self.drift).ndim > 1
                else np.asarray(self.drift).tolist()
            ),
            "true_trend": (
                np.asarray(self.trend).reshape(n, -1)[:, 0].tolist()
                if np.asarray(self.trend).ndim > 1
                else np.asarray(self.trend).tolist()
            ),
        }
        for name, mask in self.event_mask.items():
            data[f"event_{name}"] = np.asarray(mask, dtype=np.int8).tolist()
        return pl.DataFrame(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_ids": np.asarray(self.regime_ids).tolist(),
            "regime_names": list(self.regime_names),
            "volatility": np.asarray(self.volatility).tolist(),
            "drift": np.asarray(self.drift).tolist(),
            "trend": np.asarray(self.trend).tolist(),
            "transition_matrix": np.asarray(self.transition_matrix).tolist(),
            "event_mask": {k: np.asarray(v).tolist() for k, v in self.event_mask.items()},
            "metadata": dict(self.metadata),
        }


@dataclass
class SimulatedMarket:
    """Complete synthetic market dataset for one or more assets."""

    scenario_name: str
    model: str
    asset_class: str
    candles: pl.DataFrame
    trades: pl.DataFrame
    orderbook_snapshots: pl.DataFrame
    ground_truth: GroundTruth
    timestamps: list[datetime]
    symbols: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def ohlcv(self, symbol: str | None = None) -> pl.DataFrame:
        if "symbol" not in self.candles.columns:
            return self.candles
        if symbol is None:
            symbol = self.symbols[0] if self.symbols else None
        if symbol is None:
            return self.candles
        return self.candles.filter(pl.col("symbol") == symbol)

    def returns(self, symbol: str | None = None) -> np.ndarray:
        frame = self.ohlcv(symbol)
        close = frame["close"].to_numpy()
        return np.diff(np.log(np.clip(close, 1e-12, None)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "model": self.model,
            "asset_class": self.asset_class,
            "symbols": list(self.symbols),
            "n_bars": self.candles.height,
            "ground_truth": self.ground_truth.to_dict(),
            "metadata": dict(self.metadata),
        }
