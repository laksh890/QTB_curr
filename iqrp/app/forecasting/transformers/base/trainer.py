"""Low-level training loop for transformer modules."""

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


class TransformerTrainer:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.history = History()
        self.device = resolve_device(getattr(settings.train, "device", "auto"))
        self.grad_monitor = GradientMonitor()
        self._ema: dict[str, Any] | None = None

    def fit(
        self,
        module: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[Any, History]:
        if not has_torch():
            raise RuntimeError("PyTorch is required for TransformerTrainer")
        import torch

        seed_everything(self.settings.train.seed)
        module = module.to(self.device)
        if self.settings.train.gradient_checkpointing and hasattr(module, "gradient_checkpointing_enable"):
            try:
                module.gradient_checkpointing_enable()
            except Exception:  # noqa: BLE001  # pragma: no cover
                pass
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
        # curriculum: grow lookback usage via fraction of batch timesteps (soft)
        sched = build_scheduler(
            opt,
            name=self.settings.scheduler.name,
            epochs=self.settings.train.epochs,
            steps_per_epoch=max(len(loader), 1),
            warmup_epochs=self.settings.scheduler.warmup_epochs,
            min_lr=self.settings.scheduler.min_lr,
            max_lr=self.settings.train.learning_rate,
        )
        stopper = EarlyStopping(patience=self.settings.train.early_stopping_patience)
        use_amp = bool(self.settings.train.mixed_precision or self.settings.distributed.amp)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and str(self.device).startswith("cuda"))
        accum = max(int(self.settings.train.accumulation_steps), 1)
        ema_decay = float(self.settings.train.ema_decay)
        self.history = History()
        self._ema = {k: v.detach().clone() for k, v in module.state_dict().items()} if ema_decay > 0 else None

        for epoch in range(self.settings.train.epochs):
            module.train()
            losses = []
            opt.zero_grad(set_to_none=True)
            # curriculum fraction
            frac = 1.0
            if self.settings.train.curriculum:
                frac = min(1.0, 0.5 + 0.5 * (epoch + 1) / max(self.settings.train.epochs, 1))
            for step, (xb, yb) in enumerate(loader):
                if frac < 1.0 and xb.shape[1] > 4:
                    cut = max(int(xb.shape[1] * frac), 4)
                    xb = xb[:, -cut:]
                xb_t = to_tensor(xb, self.device)
                yb_t = to_tensor(yb, self.device)
                with torch.amp.autocast("cuda", enabled=use_amp and str(self.device).startswith("cuda")):
                    out = module(xb_t)
                    loss = _compute_loss(loss_fn, out, yb_t, task=self.settings.task.type) / accum
                scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    if self.settings.train.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(module.parameters(), self.settings.train.grad_clip)
                    gnorm = self.grad_monitor.record(module)
                    self.history.grad_norms.append(gnorm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if self._ema is not None:
                        with torch.no_grad():
                            for k, v in module.state_dict().items():
                                self._ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                losses.append(float(loss.item() * accum))
            train_loss = float(np.mean(losses)) if losses else 0.0
            val_loss = train_loss
            if X_val is not None and y_val is not None and X_val.size:
                val_loss = self.evaluate_loss(module, X_val, y_val, loss_fn)
            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.learning_rates.append(float(opt.param_groups[0]["lr"]))
            if sched is not None:
                if self.settings.scheduler.name == "plateau":
                    sched.step(val_loss)
                else:
                    sched.step()
            if stopper.step(val_loss, module):
                break
        if self._ema is not None:
            module.load_state_dict(self._ema)
        return module, self.history

    def evaluate_loss(self, module: Any, X: np.ndarray, y: np.ndarray, loss_fn: Any | None = None) -> float:
        import torch

        module.eval()
        loss_fn = loss_fn or get_loss(self.settings.train.loss, alphas=self.settings.task.quantile_alphas)
        with torch.no_grad():
            out = module(to_tensor(X, self.device))
            loss = _compute_loss(loss_fn, out, to_tensor(y, self.device), task=self.settings.task.type)
        return float(loss.item())

    def predict(self, module: Any, X: np.ndarray) -> np.ndarray:
        import torch

        module.eval()
        with torch.no_grad():
            # chunked / sliding inference for long context
            chunk = int(getattr(self.settings.architecture, "chunk_size", 512))
            if X.shape[1] <= chunk:
                out = module(to_tensor(X, self.device))
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return from_tensor(out)
            outs = []
            for i in range(X.shape[0]):
                xi = X[i : i + 1]
                if xi.shape[1] > chunk:
                    xi = xi[:, -chunk:]
                o = module(to_tensor(xi, self.device))
                if isinstance(o, (tuple, list)):
                    o = o[0]
                outs.append(from_tensor(o)[0])
            return np.stack(outs)


def _compute_loss(loss_fn: Any, out: Any, target: Any, *, task: str) -> Any:
    pred = out[0] if isinstance(out, (tuple, list)) else out
    if task in {"classification", "multiclass"} and pred.dim() > 1 and pred.shape[-1] > 1:
        tgt = target.long().reshape(-1)
        if pred.dim() == 3:
            b, h, c = pred.shape
            return loss_fn(pred.reshape(b * h, c), tgt.reshape(-1)[: b * h])
        return loss_fn(pred, tgt[: pred.shape[0]])
    if task in {"binary", "probability"}:
        return loss_fn(pred.reshape(-1), target.reshape(-1)[: pred.reshape(-1).shape[0]])
    if pred.shape == target.shape:
        return loss_fn(pred, target)
    if pred.dim() >= 2 and pred.shape[-1] == 1 and pred.shape[:-1] == target.shape:
        return loss_fn(pred[..., 0], target)
    if pred.shape[:-1] == target.shape:
        return loss_fn(pred[..., 0], target)
    return loss_fn(pred.reshape(target.shape[0], -1)[:, 0], target.reshape(-1)[: target.shape[0]])
