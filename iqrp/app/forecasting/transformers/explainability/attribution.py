"""Transformer explainability: attention rollout, IG, token attribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.neural.base.torch_utils import from_tensor, has_torch, to_tensor


def explain_transformer(
    module: Any,
    X: np.ndarray,
    *,
    method: str = "attention_rollout",
    device: Any = None,
) -> np.ndarray:
    method = (method or "attention_rollout").lower()
    if method in {"attention", "attention_rollout", "rollout"}:
        return attention_rollout(module, X, device=device)
    if method in {"integrated_gradients", "ig"}:
        return integrated_gradients(module, X, device=device)
    if method in {"gradient", "saliency"}:
        return saliency(module, X, device=device)
    if method == "token":
        return token_attribution(module, X, device=device)
    return attention_rollout(module, X, device=device)


def attention_rollout(module: Any, X: np.ndarray, *, device: Any = None) -> np.ndarray:
    if not has_torch():
        return np.abs(np.asarray(X, dtype=np.float64))
    import torch

    module.eval()
    with torch.no_grad():
        _ = module(to_tensor(X, device))
    attns = []
    for m in module.modules():
        if hasattr(m, "last_attn") and m.last_attn is not None:
            a = m.last_attn
            if a.dim() == 4:
                a = a.mean(dim=1)  # heads
            attns.append(a)
    Xa = np.abs(np.asarray(X, dtype=np.float64))
    if not attns:
        return Xa
    # rollout over compatible square attentions only
    result = None
    for a in attns:
        if a.size(-1) != a.size(-2):
            continue
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        t = a.size(-1)
        eye = torch.eye(t, device=a.device).unsqueeze(0).expand(a.size(0), -1, -1)
        a = 0.5 * a + 0.5 * eye
        if result is None:
            result = a
        elif result.size(-1) == a.size(-1):
            result = torch.bmm(a, result)
    if result is None:
        return Xa
    roll = from_tensor(result.mean(dim=1))  # (B, T_attn)
    # broadcast / interpolate to feature time axis
    if roll.shape[1] == Xa.shape[1]:
        return Xa * roll[:, :, None]
    # mismatch (e.g. patch tokens): use mean attention weight as scalar importance
    scale = roll.mean(axis=1, keepdims=True)[:, :, None]
    return Xa * np.maximum(scale, 1e-6)


def integrated_gradients(module: Any, X: np.ndarray, *, device: Any = None, steps: int = 12) -> np.ndarray:
    if not has_torch():
        return np.abs(np.asarray(X, dtype=np.float64))
    import torch

    module.eval()
    x = to_tensor(X, device).requires_grad_(True)
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for i in range(1, max(steps, 1) + 1):
        x_i = (baseline + (float(i) / steps) * (x - baseline)).detach().requires_grad_(True)
        out = module(x_i)
        if isinstance(out, (tuple, list)):
            out = out[0]
        scalar = out.reshape(out.shape[0], -1)[:, 0].sum()
        grads = torch.autograd.grad(scalar, x_i)[0]
        total = total + grads
    return from_tensor((x - baseline) * total / steps)


def saliency(module: Any, X: np.ndarray, *, device: Any = None) -> np.ndarray:
    if not has_torch():
        return np.abs(np.asarray(X, dtype=np.float64))
    import torch

    module.eval()
    x = to_tensor(X, device).requires_grad_(True)
    out = module(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    scalar = out.reshape(out.shape[0], -1)[:, 0].sum()
    grads = torch.autograd.grad(scalar, x)[0]
    return from_tensor(grads.abs())


def token_attribution(module: Any, X: np.ndarray, *, device: Any = None) -> np.ndarray:
    """Occlude each timestep token."""
    if not has_torch():
        return np.abs(np.asarray(X, dtype=np.float64))
    import torch

    module.eval()
    X = np.asarray(X, dtype=np.float64)
    with torch.no_grad():
        base = from_tensor(module(to_tensor(X, device)))
    if isinstance(base, list):
        base = base[0]
    base_score = np.asarray(base, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
    attr = np.zeros_like(X)
    for t in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, t, :] = 0.0
        with torch.no_grad():
            score = from_tensor(module(to_tensor(Xp, device)))
        if isinstance(score, list):
            score = score[0]
        s = np.asarray(score, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
        attr[:, t, :] = (base_score - s)[:, None]
    return attr
