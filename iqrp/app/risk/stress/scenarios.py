"""Stress scenario specifications and shock application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.risk.base import as_weights


@dataclass(slots=True)
class ScenarioSpec:
    name: str
    shocks: dict[str, float] | list[float] | np.ndarray
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def shock_vector(self, n: int, names: list[str] | None = None) -> np.ndarray:
        if isinstance(self.shocks, dict):
            out = np.zeros(n, dtype=np.float64)
            if names is not None:
                for i, nm in enumerate(names[:n]):
                    if nm in self.shocks:
                        out[i] = float(self.shocks[nm])
            else:
                for i, (_, v) in enumerate(list(self.shocks.items())[:n]):
                    out[i] = float(v)
            return out
        arr = np.asarray(self.shocks, dtype=np.float64).reshape(-1)
        out = np.zeros(n, dtype=np.float64)
        m = min(n, arr.size)
        out[:m] = arr[:m]
        return out

    def to_dict(self) -> dict[str, Any]:
        shocks: Any
        if isinstance(self.shocks, dict):
            shocks = {k: float(v) for k, v in self.shocks.items()}
        else:
            shocks = np.asarray(self.shocks, dtype=np.float64).reshape(-1).tolist()
        return {
            "name": self.name,
            "shocks": shocks,
            "description": self.description,
            "metadata": dict(self.metadata),
        }


def apply_shock(
    weights: Any,
    shocks: Any,
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """Apply additive return shocks to a portfolio: PnL = w · shock."""
    if isinstance(shocks, ScenarioSpec):
        spec = shocks
        w = as_weights(weights)
        s = spec.shock_vector(w.size, names=names)
        name = spec.name
    else:
        w = as_weights(weights)
        s = np.asarray(shocks, dtype=np.float64).reshape(-1)
        if s.size != w.size:
            tmp = np.zeros(w.size, dtype=np.float64)
            m = min(w.size, s.size)
            tmp[:m] = s[:m]
            s = tmp
        name = "custom_shock"

    pnl = float(np.dot(w, s))
    return {
        "name": "apply_shock",
        "scenario": name,
        "pnl": pnl,
        "loss": float(max(-pnl, 0.0)),
        "weights": w.tolist(),
        "shocks": s.tolist(),
    }
