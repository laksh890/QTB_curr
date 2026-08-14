"""Neural explainability: IG, gradients, occlusion, saliency."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.neural.base.torch_utils import from_tensor, has_torch, to_tensor


def explain_neural(
    module: Any,
    X: np.ndarray,
    *,
    method: str = "integrated_gradients",
    device: Any = None,
    steps: int = 16,
) -> np.ndarray:
    method = (method or "integrated_gradients").lower()
    if method in {"integrated_gradients", "ig"}:
        return integrated_gradients(module, X, device=device, steps=steps)
    if method in {"gradient", "saliency"}:
        return saliency_map(module, X, device=device)
    if method == "occlusion":
        return occlusion_analysis(module, X, device=device)
    if method == "shap":
        return _shap_or_ig(module, X, device=device)
    return integrated_gradients(module, X, device=device, steps=steps)


def integrated_gradients(
    module: Any, X: np.ndarray, *, device: Any = None, steps: int = 16
) -> np.ndarray:
    if not has_torch():
        return np.abs(np.asarray(X, dtype=np.float64))
    import torch

    module.eval()
    x = to_tensor(X, device).requires_grad_(True)
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for i in range(1, max(steps, 1) + 1):
        x_i = baseline + (float(i) / steps) * (x - baseline)
        x_i = x_i.detach().requires_grad_(True)
        out = module(x_i)
        if isinstance(out, (tuple, list)):
            out = out[0]
        scalar = out.reshape(out.shape[0], -1)[:, 0].sum()
        grads = torch.autograd.grad(scalar, x_i, retain_graph=False)[0]
        total = total + grads
    attr = (x - baseline) * total / steps
    return from_tensor(attr)


def saliency_map(module: Any, X: np.ndarray, *, device: Any = None) -> np.ndarray:
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


def occlusion_analysis(
    module: Any, X: np.ndarray, *, device: Any = None, patch: int = 2
) -> np.ndarray:
    if not has_torch():
        return np.abs(np.asarray(X, dtype=np.float64))
    import torch

    module.eval()
    X = np.asarray(X, dtype=np.float64)
    base = from_tensor(module(to_tensor(X, device)))
    if isinstance(base, list):
        base = base[0]
    base_score = np.asarray(base, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
    attr = np.zeros_like(X)
    _, t, f = X.shape
    for i in range(0, t, max(patch, 1)):
        for j in range(f):
            Xp = X.copy()
            Xp[:, i : i + patch, j] = 0.0
            with torch.no_grad():
                score = from_tensor(module(to_tensor(Xp, device)))
            if isinstance(score, list):
                score = score[0]
            s = np.asarray(score, dtype=np.float64).reshape(X.shape[0], -1)[:, 0]
            attr[:, i : i + patch, j] = (base_score - s)[:, None]
    return attr


def _shap_or_ig(module: Any, X: np.ndarray, *, device: Any = None) -> np.ndarray:
    try:
        import shap  # type: ignore

        # DeepExplainer needs background; fall back if unavailable
        background = to_tensor(X[: min(8, X.shape[0])], device)
        explainer = shap.DeepExplainer(module, background)
        vals = explainer.shap_values(to_tensor(X, device))
        if isinstance(vals, list):
            vals = vals[0]
        return np.asarray(vals, dtype=np.float64)
    except Exception:  # pragma: no cover
        return integrated_gradients(module, X, device=device)
