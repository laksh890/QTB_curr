"""Tuning utilities — search-space builders and trial bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TuningTrial:
    params: dict[str, Any]
    score: float
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"params": dict(self.params), "score": self.score, "metrics": dict(self.metrics)}


@dataclass
class TuningHistory:
    trials: list[TuningTrial] = field(default_factory=list)

    def add(self, trial: TuningTrial) -> None:
        self.trials.append(trial)

    def best(self) -> TuningTrial | None:
        if not self.trials:
            return None
        return min(self.trials, key=lambda t: t.score)

    def to_dict(self) -> dict[str, Any]:
        return {"trials": [t.to_dict() for t in self.trials]}


def build_search_space(*, family: str = "baseline") -> dict[str, list[Any]]:
    if family == "tree":
        return {"n_estimators": [25, 50], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    if family == "neural":
        return {"hidden_size": [32, 64], "num_layers": [1, 2], "learning_rate": [1e-3, 3e-4]}
    if family == "transformer":
        return {"d_model": [32, 64], "n_heads": [2, 4], "num_layers": [1, 2]}
    if family == "volatility":
        return {"p": [1, 2], "q": [1, 2]}
    if family == "statistical":
        return {"order": [(1, 0, 0), (1, 1, 1)]}
    return {"drift": [0.0, 0.01, -0.01, 0.02]}
