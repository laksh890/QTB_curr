"""Torch helpers: device selection, seeding, AMP, DDP wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


def has_torch() -> bool:
    return bool(_HAS_TORCH)


def resolve_device(preference: str = "auto") -> Any:
    if not _HAS_TORCH:
        return "cpu"
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def maybe_compile(module: Any, enabled: bool = False) -> Any:
    if not (_HAS_TORCH and enabled):
        return module
    try:
        return torch.compile(module)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001  # pragma: no cover
        return module


def count_parameters(module: Any) -> int:
    if not _HAS_TORCH or module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def to_tensor(x: np.ndarray, device: Any = None, dtype: Any = None) -> Any:
    if not _HAS_TORCH:
        return np.asarray(x, dtype=np.float64)
    dt = dtype or torch.float32
    t = torch.as_tensor(np.asarray(x), dtype=dt)
    return t.to(device) if device is not None else t


def from_tensor(x: Any) -> np.ndarray:
    if _HAS_TORCH and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
