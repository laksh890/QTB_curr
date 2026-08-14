"""Learning-rate schedulers for neural training."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        ExponentialLR,
        OneCycleLR,
        ReduceLROnPlateau,
    )
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def build_scheduler(
    optimizer: Any,
    *,
    name: str = "cosine",
    epochs: int = 30,
    steps_per_epoch: int = 10,
    warmup_epochs: int = 2,
    min_lr: float = 1e-6,
    gamma: float = 0.95,
    max_lr: float | None = None,
) -> Any | None:
    if not has_torch() or optimizer is None or name in {"none", None}:
        return None
    key = str(name).lower()
    if key == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    if key == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=max_lr or float(optimizer.param_groups[0]["lr"]),
            epochs=max(epochs, 1),
            steps_per_epoch=max(steps_per_epoch, 1),
        )
    if key == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=min_lr)
    if key == "exponential":
        return ExponentialLR(optimizer, gamma=gamma)
    if key == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, min_lr=min_lr
        )
    return CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)


class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(
        self,
        optimizer: Any,
        *,
        warmup_epochs: int = 2,
        total_epochs: int = 30,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.total_epochs = max(int(total_epochs), 1)
        self.min_lr = float(min_lr)
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.last_epoch = -1

    def step(self, metrics: float | None = None) -> None:
        self.last_epoch += 1
        e = self.last_epoch
        for i, group in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i]
            if e < self.warmup_epochs:
                lr = base * float(e + 1) / max(self.warmup_epochs, 1)
            else:
                import math

                progress = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * progress))
            group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return [float(g["lr"]) for g in self.optimizer.param_groups]


def build_optimizer(
    parameters: Any,
    *,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Any:
    if not has_torch():
        return None
    import torch

    key = str(name).lower()
    params = list(parameters)
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if key == "lion":
        return Lion(params, lr=lr, weight_decay=weight_decay)
    if key == "lookahead":
        base = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return Lookahead(base)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


class Lion(torch.optim.Optimizer if has_torch() else object):  # type: ignore[misc]
    """Simple Lion optimizer (Chen et al.)."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        if has_torch():
            super().__init__(params, defaults)

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            lr = g["lr"]
            beta1, beta2 = g["betas"]
            wd = g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                state = self.state.setdefault(p, {})
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p.data)
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                update = exp_avg.sign()
                p.data.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class Lookahead:
    """Lookahead wrapper around an inner optimizer (scheduler-compatible)."""

    def __init__(self, optimizer: Any, k: int = 5, alpha: float = 0.5) -> None:
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups
        self.state = getattr(optimizer, "state", {})
        self.defaults = getattr(optimizer, "defaults", {})
        self._step_count = 0
        self._backup = [[p.data.clone() for p in g["params"]] for g in self.param_groups]

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> Any:
        loss = self.optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for g_idx, g in enumerate(self.param_groups):
                for p_idx, p in enumerate(g["params"]):
                    slow = self._backup[g_idx][p_idx]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    p.data.copy_(slow)
        return loss
