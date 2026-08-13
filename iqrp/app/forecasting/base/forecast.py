"""Unified multi-horizon forecast container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.forecasting.base.prediction import (
    DistributionForecast,
    Prediction,
    PredictionInterval,
    QuantileForecast,
)


@dataclass(slots=True)
class Forecast:
    """Canonical forecast object consumed by downstream IQRP modules.

    Supports point paths, intervals, quantiles, and optional distributions
    across one or more horizons.
    """

    values: np.ndarray
    horizon: int
    timestamps: tuple[Any, ...] = ()
    intervals: list[PredictionInterval] | None = None
    confidence_intervals: list[PredictionInterval] | None = None
    probabilities: np.ndarray | None = None
    quantiles: list[QuantileForecast] | None = None
    distribution: DistributionForecast | None = None
    features_used: tuple[str, ...] = ()
    regime_used: int | str | np.ndarray | None = None
    model_name: str = ""
    model_version: str = ""
    strategy: str = "direct"  # recursive | direct | sequence | multi_step
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim == 0:
            self.values = self.values.reshape(1)
        if self.probabilities is not None:
            self.probabilities = np.asarray(self.probabilities, dtype=np.float64)

    @classmethod
    def from_values(
        cls,
        values: np.ndarray | list[float],
        *,
        horizon: int | None = None,
        timestamps: tuple[Any, ...] | list[Any] | None = None,
        model_name: str = "",
        model_version: str = "",
        features_used: tuple[str, ...] = (),
        regime_used: Any = None,
        strategy: str = "direct",
        metadata: dict[str, Any] | None = None,
        probabilities: np.ndarray | None = None,
        intervals: list[PredictionInterval] | None = None,
        confidence_intervals: list[PredictionInterval] | None = None,
    ) -> Forecast:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        h = int(horizon if horizon is not None else arr.size)
        return cls(
            values=arr,
            horizon=h,
            timestamps=tuple(timestamps or ()),
            intervals=intervals,
            confidence_intervals=confidence_intervals,
            probabilities=probabilities,
            features_used=features_used,
            regime_used=regime_used,
            model_name=model_name,
            model_version=model_version,
            strategy=strategy,
            metadata=dict(metadata or {}),
        )

    def point(self, step: int = 1) -> Prediction:
        idx = max(int(step) - 1, 0)
        if idx >= self.values.size:
            raise IndexError(f"horizon step {step} exceeds forecast length {self.values.size}")
        ts = self.timestamps[idx] if idx < len(self.timestamps) else None
        prob = None
        if self.probabilities is not None:
            prob = self.probabilities[idx] if self.probabilities.ndim > 1 else self.probabilities
        return Prediction(
            value=float(self.values[idx]),
            timestamp=ts,
            horizon=idx + 1,
            probability=prob,
            regime=self._regime_at(idx),
            features_used=self.features_used,
            model_version=self.model_version,
            metadata={"model_name": self.model_name},
        )

    def one_step(self) -> Prediction:
        return self.point(1)

    def n_step(self, n: int) -> Prediction:
        return self.point(n)

    def path(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    def _regime_at(self, idx: int) -> int | str | None:
        r = self.regime_used
        if r is None:
            return None
        if isinstance(r, np.ndarray):
            if r.size == 0:
                return None
            return r[min(idx, r.size - 1)].item() if hasattr(r[min(idx, r.size - 1)], "item") else r[min(idx, r.size - 1)]
        return r

    def to_dict(self) -> dict[str, Any]:
        regime = self.regime_used
        if isinstance(regime, np.ndarray):
            regime = regime.tolist()
        return {
            "values": self.values.tolist(),
            "horizon": self.horizon,
            "timestamps": list(self.timestamps),
            "intervals": None if self.intervals is None else [i.to_dict() for i in self.intervals],
            "confidence_intervals": (
                None
                if self.confidence_intervals is None
                else [i.to_dict() for i in self.confidence_intervals]
            ),
            "probabilities": None if self.probabilities is None else self.probabilities.tolist(),
            "quantiles": None if self.quantiles is None else [q.to_dict() for q in self.quantiles],
            "distribution": None if self.distribution is None else self.distribution.to_dict(),
            "features_used": list(self.features_used),
            "regime_used": regime,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "strategy": self.strategy,
            "metadata": dict(self.metadata),
        }
