"""Mixture-of-Experts components for regime-adaptive transformers."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class ExpertFFN(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


class MoERouter(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int, n_experts: int, top_k: int = 2) -> None:
        if has_torch():
            super().__init__()
        self.gate = nn.Linear(d_model, max(int(n_experts), 1))
        self.top_k = max(int(top_k), 1)
        self.n_experts = max(int(n_experts), 1)

    def forward(self, x: Any) -> tuple[Any, Any]:
        # x: (B, T, D) or (B, D)
        logits = self.gate(x)
        weights = torch.softmax(logits, dim=-1)
        top = torch.topk(weights, k=min(self.top_k, self.n_experts), dim=-1)
        return weights, top


class MoEGating(nn.Module if has_torch() else object):  # type: ignore[misc]
    """Soft mixture over experts (dense gating for stability on CPU)."""

    def __init__(self, d_model: int, n_experts: int, ffn_dim: int, dropout: float = 0.1) -> None:
        if has_torch():
            super().__init__()
        self.router = MoERouter(d_model, n_experts, top_k=2)
        self.experts = nn.ModuleList([ExpertFFN(d_model, ffn_dim, dropout) for _ in range(max(n_experts, 1))])

    def forward(self, x: Any, regime_ids: Any | None = None) -> Any:
        weights, _ = self.router(x)
        # dense mix for correctness / coverage
        outs = torch.stack([e(x) for e in self.experts], dim=-1)  # (..., D, E)
        w = weights.unsqueeze(-2)  # (..., 1, E)
        return (outs * w).sum(dim=-1)


__all__ = ["ExpertFFN", "MoERouter", "MoEGating"]
