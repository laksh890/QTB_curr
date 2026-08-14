"""TargetWeights dataclass and builders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class TargetWeights:
    """Named target portfolio weights with audit metadata."""

    names: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    method: str = ""
    source: str = "optimization"
    long_only: bool = True
    budget: float = 1.0
    timestamp: str = field(default_factory=_utc_now)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.names) and self.names:
            w = list(self.weights)
            if len(w) < len(self.names):
                w.extend([0.0] * (len(self.names) - len(w)))
            self.weights = w[: len(self.names)]

    @property
    def n_assets(self) -> int:
        return len(self.names)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return {n: float(w) for n, w in zip(self.names, self.weights)}

    def gross_exposure(self) -> float:
        return float(np.sum(np.abs(self.as_array())))

    def net_exposure(self) -> float:
        return float(np.sum(self.as_array()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "weights": [float(w) for w in self.weights],
            "method": self.method,
            "source": self.source,
            "long_only": bool(self.long_only),
            "budget": float(self.budget),
            "timestamp": self.timestamp,
            "meta": dict(self.meta),
            "gross_exposure": self.gross_exposure(),
            "net_exposure": self.net_exposure(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetWeights:
        return cls(
            names=list(data.get("names") or []),
            weights=[float(w) for w in (data.get("weights") or [])],
            method=str(data.get("method", "")),
            source=str(data.get("source", "optimization")),
            long_only=bool(data.get("long_only", True)),
            budget=float(data.get("budget", 1.0)),
            timestamp=str(data.get("timestamp", _utc_now())),
            meta=dict(data.get("meta") or {}),
        )

    @classmethod
    def from_arrays(
        cls,
        weights: Sequence[float] | np.ndarray,
        *,
        names: Sequence[str] | None = None,
        method: str = "",
        source: str = "optimization",
        long_only: bool = True,
        budget: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> TargetWeights:
        w = [float(x) for x in np.asarray(weights, dtype=np.float64).reshape(-1).tolist()]
        n = len(w)
        name_list = list(names) if names is not None else [f"a{i}" for i in range(n)]
        if len(name_list) != n:
            name_list = [f"a{i}" for i in range(n)]
        bud = float(budget) if budget is not None else float(np.sum(w))
        return cls(
            names=name_list,
            weights=w,
            method=method,
            source=source,
            long_only=long_only,
            budget=bud,
            meta=dict(meta or {}),
        )

    @classmethod
    def cash(cls, *, currency: str = "USD") -> TargetWeights:
        return cls(
            names=[],
            weights=[],
            method="cash",
            source="fallback",
            long_only=True,
            budget=0.0,
            meta={"currency": currency, "fallback": "cash"},
        )

    @classmethod
    def equal_weight(
        cls,
        n: int | Sequence[str],
        *,
        budget: float = 1.0,
        method: str = "equal_weight",
    ) -> TargetWeights:
        if isinstance(n, int):
            names = [f"a{i}" for i in range(n)]
            k = n
        else:
            names = list(n)
            k = len(names)
        if k <= 0:
            return cls.cash()
        w = [float(budget) / k] * k
        return cls(names=names, weights=w, method=method, source="builder", budget=float(budget))


def build_target_weights(
    weights: Sequence[float] | np.ndarray | dict[str, float],
    *,
    names: Sequence[str] | None = None,
    method: str = "",
    source: str = "optimization",
    long_only: bool = True,
    meta: dict[str, Any] | None = None,
) -> TargetWeights:
    """Build TargetWeights from list, array, or name→weight mapping."""
    if isinstance(weights, dict):
        name_list = list(names) if names is not None else list(weights.keys())
        w = [float(weights.get(nm, 0.0)) for nm in name_list]
        return TargetWeights.from_arrays(
            w, names=name_list, method=method, source=source, long_only=long_only, meta=meta
        )
    return TargetWeights.from_arrays(
        weights, names=names, method=method, source=source, long_only=long_only, meta=meta
    )
