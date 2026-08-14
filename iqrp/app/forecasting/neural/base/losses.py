"""Neural forecasting loss functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


_CUSTOM: dict[str, Callable[..., Any]] = {}


def register_custom_loss(name: str, fn: Callable[..., Any]) -> None:
    _CUSTOM[name] = fn


def get_loss(
    name: str, *, alphas: tuple[float, ...] | None = None, label_smoothing: float = 0.0
) -> Any:
    if not has_torch():
        return _NumpyLoss(name, alphas=alphas)
    key = name.lower()
    if key in _CUSTOM:
        return _CUSTOM[key]
    if key == "mse":
        return nn.MSELoss()
    if key == "mae":
        return nn.L1Loss()
    if key == "huber":
        return nn.SmoothL1Loss()
    if key == "logcosh":
        return LogCoshLoss()
    if key == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if key == "bce":
        return nn.BCEWithLogitsLoss()
    if key == "focal":
        return FocalLoss()
    if key == "quantile":
        return QuantileLoss(alphas=alphas or (0.1, 0.5, 0.9))
    if key == "gaussian_nll":
        return GaussianNLLLoss()
    if key == "student_t_nll":
        return StudentTNLLLoss()
    return nn.MSELoss()


class LogCoshLoss(nn.Module if has_torch() else object):  # type: ignore[misc]
    def forward(self, pred: Any, target: Any) -> Any:
        err = pred - target
        return torch.mean(torch.log(torch.cosh(err + 1e-12)))


class FocalLoss(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        if has_torch():
            super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Any, target: Any) -> Any:
        # binary focal on logits
        prob = torch.sigmoid(logits)
        t = target.float()
        pt = prob * t + (1 - prob) * (1 - t)
        w = self.alpha * t + (1 - self.alpha) * (1 - t)
        loss = -w * (1 - pt) ** self.gamma * torch.log(pt.clamp(1e-8, 1 - 1e-8))
        return loss.mean()


class QuantileLoss(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)) -> None:
        if has_torch():
            super().__init__()
        self.alphas = alphas

    def forward(self, pred: Any, target: Any) -> Any:
        # pred: (B, H, Q) or (B, Q)
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        losses = []
        for i, a in enumerate(self.alphas):
            e = target.squeeze(-1) - pred[..., i]
            losses.append(torch.maximum(a * e, (a - 1) * e).mean())
        return sum(losses) / len(losses)


class GaussianNLLLoss(nn.Module if has_torch() else object):  # type: ignore[misc]
    def forward(self, pred: Any, target: Any) -> Any:
        # pred last dim: [mu, log_sigma]
        if pred.shape[-1] < 2:
            return F.mse_loss(pred.squeeze(-1), target)
        mu, log_sigma = pred[..., 0], pred[..., 1]
        var = torch.exp(log_sigma * 2).clamp(1e-6, 1e6)
        return F.gaussian_nll_loss(mu, target, var, full=True)


class StudentTNLLLoss(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, df: float = 5.0) -> None:
        if has_torch():
            super().__init__()
        self.df = df

    def forward(self, pred: Any, target: Any) -> Any:
        if pred.shape[-1] < 2:
            return F.mse_loss(pred.squeeze(-1), target)
        mu, log_scale = pred[..., 0], pred[..., 1]
        scale = torch.exp(log_scale).clamp(1e-6, 1e6)
        z = (target - mu) / scale
        # student-t nll up to constants
        return torch.mean(log_scale + 0.5 * (self.df + 1) * torch.log1p(z**2 / self.df))


class _NumpyLoss:
    def __init__(self, name: str, alphas: tuple[float, ...] | None = None) -> None:
        self.name = name
        self.alphas = alphas or (0.5,)

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        if self.name == "mae":
            return float(np.mean(np.abs(p - t)))
        return float(np.mean((p - t) ** 2))
