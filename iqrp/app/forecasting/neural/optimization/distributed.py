"""Distributed training helpers (single-process / optional DDP stubs)."""

from __future__ import annotations

from typing import Any


def wrap_ddp(module: Any, *, enabled: bool = False) -> Any:
    """Optionally wrap module in DistributedDataParallel when world size > 1."""
    if not enabled:
        return module
    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not dist.is_available() or not dist.is_initialized():
            return module
        if dist.get_world_size() <= 1:
            return module
        device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
        return DDP(module, device_ids=device_ids)
    except Exception:  # noqa: BLE001  # pragma: no cover
        return module


def enable_gradient_checkpointing(module: Any) -> Any:
    if hasattr(module, "gradient_checkpointing_enable"):
        try:
            module.gradient_checkpointing_enable()
        except Exception:  # noqa: BLE001  # pragma: no cover
            pass
    return module


def amp_enabled(settings: Any) -> bool:
    train = getattr(settings, "train", None)
    dist = getattr(settings, "distributed", None)
    return bool(
        (train is not None and getattr(train, "mixed_precision", False))
        or (dist is not None and getattr(dist, "amp", False))
    )
