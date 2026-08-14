"""Training callbacks: early stopping, checkpointing, history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": list(self.train_loss),
            "val_loss": list(self.val_loss),
            "learning_rates": list(self.learning_rates),
            "grad_norms": list(self.grad_norms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> History:
        return cls(
            train_loss=list(data.get("train_loss") or []),
            val_loss=list(data.get("val_loss") or []),
            learning_rates=list(data.get("learning_rates") or []),
            grad_norms=list(data.get("grad_norms") or []),
        )


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-6) -> None:
        self.patience = max(int(patience), 1)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_epochs = 0
        self.should_stop = False
        self.best_state: dict[str, Any] | None = None

    def step(self, metric: float, model: Any | None = None) -> bool:
        if self.best is None or metric < self.best - self.min_delta:
            self.best = float(metric)
            self.bad_epochs = 0
            if model is not None and hasattr(model, "state_dict"):
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            return False
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
            if (
                self.best_state is not None
                and model is not None
                and hasattr(model, "load_state_dict")
            ):
                model.load_state_dict(self.best_state)
            return True
        return False


class GradientMonitor:
    def __init__(self) -> None:
        self.norms: list[float] = []

    def record(self, model: Any) -> float:
        total = 0.0
        if hasattr(model, "parameters"):
            for p in model.parameters():
                if p.grad is not None:
                    total += float(p.grad.data.norm(2).item() ** 2)
        norm = total**0.5
        self.norms.append(norm)
        return norm
