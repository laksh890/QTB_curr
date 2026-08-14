"""Core training loop for neural forecasting models."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.neural.base.callbacks import EarlyStopping, GradientMonitor, History
from iqrp.app.forecasting.neural.base.data import NumpyBatchLoader
from iqrp.app.forecasting.neural.base.losses import get_loss
from iqrp.app.forecasting.neural.base.scheduler import build_optimizer, build_scheduler
from iqrp.app.forecasting.neural.base.torch_utils import (
    from_tensor,
    has_torch,
    maybe_compile,
    resolve_device,
    seed_everything,
    to_tensor,
)


class NeuralTrainer:
    """Fits a torch Module with AMP / clipping / early stopping / schedulers."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.history = History()
        self.device = resolve_device(getattr(settings.train, "device", "auto"))
        self.grad_monitor = GradientMonitor()

    def fit(
        self,
        module: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[Any, History]:
        if not has_torch():
            raise RuntimeError("PyTorch is required for NeuralTrainer")
        import torch

        seed_everything(self.settings.train.seed)
        module = module.to(self.device)
        module = maybe_compile(module, self.settings.train.compile)
        opt = build_optimizer(
            module.parameters(),
            name=self.settings.train.optimizer,
            lr=self.settings.train.learning_rate,
            weight_decay=self.settings.train.weight_decay,
        )
        loss_fn = get_loss(
            self.settings.train.loss,
            alphas=self.settings.task.quantile_alphas,
            label_smoothing=self.settings.train.label_smoothing,
        )
        loader = NumpyBatchLoader(
            X_train,
            y_train,
            batch_size=self.settings.train.batch_size,
            shuffle=True,
            seed=self.settings.train.seed,
        )
        sched = build_scheduler(
            opt,
            name=self.settings.scheduler.name,
            epochs=self.settings.train.epochs,
            steps_per_epoch=max(len(loader), 1),
            warmup_epochs=self.settings.scheduler.warmup_epochs,
            min_lr=self.settings.scheduler.min_lr,
            gamma=self.settings.scheduler.gamma,
            max_lr=self.settings.train.learning_rate,
        )
        stopper = EarlyStopping(patience=self.settings.train.early_stopping_patience)
        use_amp = bool(self.settings.train.mixed_precision or self.settings.distributed.amp)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_amp and str(self.device).startswith("cuda")
        )
        accum = max(int(self.settings.train.accumulation_steps), 1)
        self.history = History()

        for epoch in range(self.settings.train.epochs):
            module.train()
            losses = []
            opt.zero_grad(set_to_none=True)
            for step, (xb, yb) in enumerate(loader):
                xb_t = to_tensor(xb, self.device)
                yb_t = to_tensor(yb, self.device)
                with torch.amp.autocast(
                    "cuda", enabled=use_amp and str(self.device).startswith("cuda")
                ):
                    out = module(xb_t)
                    loss = _compute_loss(loss_fn, out, yb_t, task=self.settings.task.type) / accum
                scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    if self.settings.train.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(), self.settings.train.grad_clip
                        )
                    gnorm = self.grad_monitor.record(module)
                    self.history.grad_norms.append(gnorm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if sched is not None and self.settings.scheduler.name == "onecycle":
                        sched.step()
                losses.append(float(loss.item() * accum))
            train_loss = float(np.mean(losses)) if losses else 0.0
            val_loss = train_loss
            if X_val is not None and y_val is not None and X_val.size:
                val_loss = self.evaluate_loss(module, X_val, y_val, loss_fn)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            lr = float(opt.param_groups[0]["lr"])
            self.history.learning_rates.append(lr)
            if sched is not None and self.settings.scheduler.name != "onecycle":
                if self.settings.scheduler.name == "plateau":
                    sched.step(val_loss)
                else:
                    sched.step()
            if stopper.step(val_loss, module):
                break
        return module, self.history

    def evaluate_loss(
        self, module: Any, X: np.ndarray, y: np.ndarray, loss_fn: Any | None = None
    ) -> float:
        import torch

        module.eval()
        loss_fn = loss_fn or get_loss(
            self.settings.train.loss, alphas=self.settings.task.quantile_alphas
        )
        with torch.no_grad():
            xb = to_tensor(X, self.device)
            yb = to_tensor(y, self.device)
            out = module(xb)
            loss = _compute_loss(loss_fn, out, yb, task=self.settings.task.type)
        return float(loss.item())

    def predict(self, module: Any, X: np.ndarray) -> np.ndarray:
        import torch

        module.eval()
        with torch.no_grad():
            out = module(to_tensor(X, self.device))
            if isinstance(out, (tuple, list)):
                out = out[0]
            return from_tensor(out)


def _compute_loss(loss_fn: Any, out: Any, target: Any, *, task: str) -> Any:

    pred = out[0] if isinstance(out, (tuple, list)) else out
    if task in {"classification", "multiclass"} and pred.dim() > 1 and pred.shape[-1] > 1:
        tgt = target.long().reshape(-1)
        if pred.dim() == 3:
            # (B,H,C) -> flatten
            b, h, c = pred.shape
            return loss_fn(pred.reshape(b * h, c), tgt.reshape(-1)[: b * h])
        return loss_fn(pred, tgt[: pred.shape[0]])
    if task in {"binary", "probability"}:
        return loss_fn(pred.reshape(-1), target.reshape(-1)[: pred.reshape(-1).shape[0]])
    # regression / quantile / distribution
    if pred.shape == target.shape:
        return loss_fn(pred, target)
    if (
        pred.dim() >= 2
        and target.dim() == 2
        and pred.shape[:2] == target.shape[:2]
        and pred.shape == (*target.shape, pred.shape[-1])
        and pred.shape[-1] > 1
    ):
        return loss_fn(pred, target)
    # squeeze trailing singleton
    if pred.dim() >= 2 and pred.shape[-1] == 1 and pred.shape[:-1] == target.shape:
        return loss_fn(pred[..., 0], target)
    if pred.shape[:-1] == target.shape:
        return loss_fn(pred[..., 0], target)
    return loss_fn(pred.reshape(target.shape[0], -1)[:, 0], target.reshape(-1)[: target.shape[0]])
