"""Forecasting model metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ForecastTask = Literal[
    "regression",
    "classification",
    "probability",
    "multi_step",
    "recursive",
    "direct",
    "sequence",
]


@dataclass(frozen=True, slots=True)
class ForecastModelMeta:
    """Immutable identity for a registered forecasting algorithm."""

    name: str
    version: str
    description: str
    algorithm_family: str
    task: ForecastTask = "regression"
    default_horizon: int = 1
    supports_online: bool = False
    supports_proba: bool = False
    supports_intervals: bool = False
    supports_quantiles: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    target_names: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "algorithm_family": self.algorithm_family,
            "task": self.task,
            "default_horizon": self.default_horizon,
            "supports_online": self.supports_online,
            "supports_proba": self.supports_proba,
            "supports_intervals": self.supports_intervals,
            "supports_quantiles": self.supports_quantiles,
            "parameters": dict(self.parameters),
            "target_names": list(self.target_names),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastModelMeta:
        return cls(
            name=str(data["name"]),
            version=str(data.get("version", "0.0.0")),
            description=str(data.get("description", "")),
            algorithm_family=str(data.get("algorithm_family", "unknown")),
            task=data.get("task", "regression"),  # type: ignore[arg-type]
            default_horizon=int(data.get("default_horizon", 1)),
            supports_online=bool(data.get("supports_online", False)),
            supports_proba=bool(data.get("supports_proba", False)),
            supports_intervals=bool(data.get("supports_intervals", False)),
            supports_quantiles=bool(data.get("supports_quantiles", False)),
            parameters=dict(data.get("parameters") or {}),
            target_names=tuple(data.get("target_names") or ()),
            feature_names=tuple(data.get("feature_names") or ()),
        )


@dataclass(frozen=True, slots=True)
class TrainingMetadata:
    """Captured at fit time for registry / audit trails."""

    n_samples: int
    n_features: int
    feature_columns: tuple[str, ...]
    target_column: str | None
    regime_column: str | None
    horizon: int
    train_start: Any = None
    train_end: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "regime_column": self.regime_column,
            "horizon": self.horizon,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingMetadata:
        return cls(
            n_samples=int(data.get("n_samples", 0)),
            n_features=int(data.get("n_features", 0)),
            feature_columns=tuple(data.get("feature_columns") or ()),
            target_column=data.get("target_column"),
            regime_column=data.get("regime_column"),
            horizon=int(data.get("horizon", 1)),
            train_start=data.get("train_start"),
            train_end=data.get("train_end"),
            extra=dict(data.get("extra") or {}),
        )
