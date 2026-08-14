"""Categorical and regime embeddings."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class CategoricalEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_categories: int, dim: int = 16) -> None:
        if has_torch():
            super().__init__()
            self.emb = nn.Embedding(max(int(n_categories), 1), dim)

    def forward(self, x: Any) -> Any:
        return self.emb(x.long().clamp(min=0))


class RegimeEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_regimes: int = 4, dim: int = 16) -> None:
        if has_torch():
            super().__init__()
            self.emb = nn.Embedding(max(int(n_regimes), 1), dim)

    def forward(self, regime_ids: Any) -> Any:
        return self.emb(regime_ids.long().clamp(min=0))


class RegimeGate(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, hidden: int, n_regimes: int = 4) -> None:
        if has_torch():
            super().__init__()
            self.gate = nn.Linear(n_regimes, hidden)

    def forward(self, h: Any, regime_onehot: Any) -> Any:
        g = torch.sigmoid(self.gate(regime_onehot))
        return h * g


class MixtureOfExperts(nn.Module if has_torch() else object):  # type: ignore[misc]
    """Soft regime-conditioned mixture-of-experts routing over expert outputs."""

    def __init__(self, n_experts: int, hidden: int) -> None:
        if has_torch():
            super().__init__()
            self.router = nn.Linear(hidden, max(int(n_experts), 1))
            self.n_experts = max(int(n_experts), 1)

    def forward(self, h: Any, expert_outputs: Any) -> Any:
        # h: (B, H), expert_outputs: (B, E, ...) or list of (B, ...)
        if isinstance(expert_outputs, (list, tuple)):
            stacked = torch.stack(list(expert_outputs), dim=1)
        else:
            stacked = expert_outputs
        weights = torch.softmax(self.router(h), dim=-1)  # (B, E)
        while weights.dim() < stacked.dim():
            weights = weights.unsqueeze(-1)
        return (weights * stacked).sum(dim=1)
