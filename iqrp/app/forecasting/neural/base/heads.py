"""Shared neural architecture building blocks."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


def output_head(
    in_dim: int,
    horizon: int,
    *,
    task: str,
    n_classes: int = 2,
    n_quantiles: int = 3,
    dist: bool = False,
) -> Any:
    if task in {"classification", "multiclass"}:
        return nn.Linear(in_dim, horizon * n_classes)
    if task in {"binary", "probability"}:
        return nn.Linear(in_dim, horizon)
    if task == "quantile":
        return nn.Linear(in_dim, horizon * n_quantiles)
    if task == "distribution" or dist:
        return nn.Linear(in_dim, horizon * 2)  # mu, log_sigma
    return nn.Linear(in_dim, horizon)


def reshape_head(
    out: Any,
    batch: int,
    horizon: int,
    *,
    task: str,
    n_classes: int = 2,
    n_quantiles: int = 3,
    dist: bool = False,
) -> Any:
    if task in {"classification", "multiclass"}:
        return out.view(batch, horizon, n_classes)
    if task == "quantile":
        return out.view(batch, horizon, n_quantiles)
    if task == "distribution" or dist:
        return out.view(batch, horizon, 2)
    return out.view(batch, horizon)
