"""Multi-modal embeddings for transformer forecasting."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class ValueEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_features: int, d_model: int) -> None:
        if has_torch():
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)

    def forward(self, x: Any) -> Any:
        return self.proj(x)


class TimeEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int) -> None:
        if has_torch():
            super().__init__()
            self.proj = nn.Linear(4, d_model)

    def forward(self, t_feat: Any) -> Any:
        if t_feat.shape[-1] != 4:
            pad = torch.zeros(*t_feat.shape[:-1], 4, device=t_feat.device, dtype=t_feat.dtype)
            n = min(t_feat.shape[-1], 4)
            pad[..., :n] = t_feat[..., :n]
            t_feat = pad
        return self.proj(t_feat)


class AssetEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_assets: int, d_model: int) -> None:
        if has_torch():
            super().__init__()
            self.emb = nn.Embedding(max(int(n_assets), 1), d_model)

    def forward(self, asset_ids: Any) -> Any:
        return self.emb(asset_ids.long().clamp(min=0))


class RegimeEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_regimes: int, d_model: int) -> None:
        if has_torch():
            super().__init__()
            self.emb = nn.Embedding(max(int(n_regimes), 1), d_model)

    def forward(self, regime_ids: Any) -> Any:
        return self.emb(regime_ids.long().clamp(min=0))


class SectorEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_sectors: int, d_model: int) -> None:
        if has_torch():
            super().__init__()
            self.emb = nn.Embedding(max(int(n_sectors), 1), d_model)

    def forward(self, sector_ids: Any) -> Any:
        return self.emb(sector_ids.long().clamp(min=0))


class CalendarEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, d_model: int) -> None:
        if has_torch():
            super().__init__()
            self.dow = nn.Embedding(7, d_model)
            self.month = nn.Embedding(12, d_model)

    def forward(self, dow: Any, month: Any) -> Any:
        return self.dow(dow.long().clamp(0, 6)) + self.month(month.long().clamp(0, 11))


class CategoricalEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, n_categories: int, d_model: int) -> None:
        if has_torch():
            super().__init__()
            self.emb = nn.Embedding(max(int(n_categories), 1), d_model)

    def forward(self, ids: Any) -> Any:
        return self.emb(ids.long().clamp(min=0))


class TransformerInputEmbedding(nn.Module if has_torch() else object):  # type: ignore[misc]
    """Compose value + optional regime / asset embeddings."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        *,
        n_regimes: int = 4,
        n_assets: int = 1,
        use_regime: bool = True,
        dropout: float = 0.1,
    ) -> None:
        if has_torch():
            super().__init__()
        self.value = ValueEmbedding(n_features, d_model)
        self.regime = RegimeEmbedding(n_regimes, d_model) if use_regime else None
        self.asset = AssetEmbedding(n_assets, d_model) if n_assets > 1 else None
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Any,
        *,
        regime_ids: Any | None = None,
        asset_ids: Any | None = None,
    ) -> Any:
        h = self.value(x)
        if self.regime is not None and regime_ids is not None:
            if regime_ids.dim() == 1:
                regime_ids = regime_ids.unsqueeze(-1).expand(-1, x.size(1))
            h = h + self.regime(regime_ids)
        if self.asset is not None and asset_ids is not None:
            if asset_ids.dim() == 1:
                asset_ids = asset_ids.unsqueeze(-1).expand(-1, x.size(1))
            h = h + self.asset(asset_ids)
        return self.drop(self.norm(h))
