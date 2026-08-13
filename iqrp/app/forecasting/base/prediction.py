"""Atomic prediction containers for forecasting outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Prediction:
    """Point prediction for a single timestamp / horizon step."""

    value: float | np.ndarray
    timestamp: Any = None
    horizon: int = 1
    probability: float | np.ndarray | None = None
    class_id: int | None = None
    regime: int | str | None = None
    features_used: tuple[str, ...] = ()
    model_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        val = self.value
        if isinstance(val, np.ndarray):
            val = val.tolist()
        prob = self.probability
        if isinstance(prob, np.ndarray):
            prob = prob.tolist()
        return {
            "value": val,
            "timestamp": self.timestamp,
            "horizon": self.horizon,
            "probability": prob,
            "class_id": self.class_id,
            "regime": self.regime,
            "features_used": list(self.features_used),
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class PredictionInterval:
    """Symmetric or asymmetric prediction / confidence interval."""

    lower: float | np.ndarray
    upper: float | np.ndarray
    level: float = 0.95
    kind: str = "prediction"  # prediction | confidence
    midpoint: float | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def _ser(x: float | np.ndarray | None) -> Any:
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x.tolist()
            return float(x)

        return {
            "lower": _ser(self.lower),
            "upper": _ser(self.upper),
            "level": self.level,
            "kind": self.kind,
            "midpoint": _ser(self.midpoint),
            "metadata": dict(self.metadata),
        }

    @property
    def width(self) -> float | np.ndarray:
        return np.asarray(self.upper, dtype=np.float64) - np.asarray(self.lower, dtype=np.float64)


@dataclass(slots=True)
class QuantileForecast:
    """Multi-quantile forecast at a fixed horizon."""

    quantiles: dict[float, float | np.ndarray]
    horizon: int = 1
    timestamp: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        q = {}
        for k, v in self.quantiles.items():
            q[str(k)] = v.tolist() if isinstance(v, np.ndarray) else float(v)
        return {
            "quantiles": q,
            "horizon": self.horizon,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class DistributionForecast:
    """Parametric or sample-based predictive distribution."""

    mean: float | np.ndarray
    variance: float | np.ndarray | None = None
    samples: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)
    horizon: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist() if isinstance(self.mean, np.ndarray) else float(self.mean),
            "variance": (
                None
                if self.variance is None
                else (
                    self.variance.tolist()
                    if isinstance(self.variance, np.ndarray)
                    else float(self.variance)
                )
            ),
            "samples": None if self.samples is None else np.asarray(self.samples).tolist(),
            "params": dict(self.params),
            "horizon": self.horizon,
            "metadata": dict(self.metadata),
        }
