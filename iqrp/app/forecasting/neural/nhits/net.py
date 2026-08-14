"""N-HiTS backbone with hierarchical interpolation stacks."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class NHitsBlock(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, lookback: int, horizon: int, hidden: int, pool_size: int = 1) -> None:
        if has_torch():
            super().__init__()
        self.pool_size = max(int(pool_size), 1)
        self.fc = nn.Sequential(
            nn.Linear(lookback, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden, lookback)
        self.forecast = nn.Linear(hidden, max(horizon // self.pool_size, 1))
        self.horizon = horizon

    def forward(self, x: Any) -> tuple[Any, Any]:
        # x: (B, L)
        if self.pool_size > 1:
            # average pool along time by reshape
            b, l = x.shape
            pad = (self.pool_size - l % self.pool_size) % self.pool_size
            if pad:
                x = F.pad(x, (0, pad))
            x_p = x.view(b, -1, self.pool_size).mean(-1)
            # upsample back to lookback via interpolate
            x_p = F.interpolate(
                x_p.unsqueeze(1), size=l, mode="linear", align_corners=False
            ).squeeze(1)
        else:
            x_p = x
        h = self.fc(x_p)
        backcast = self.backcast(h)
        f = self.forecast(h)
        if f.shape[-1] != self.horizon:
            f = F.interpolate(
                f.unsqueeze(1), size=self.horizon, mode="linear", align_corners=False
            ).squeeze(1)
        return backcast, f


class NHitsNet(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        lookback: int,
        horizon: int,
        *,
        hidden_size: int = 64,
        n_blocks: int = 3,
        task: str = "regression",
        n_quantiles: int = 3,
        dist: bool = False,
    ) -> None:
        if has_torch():
            super().__init__()
        self.horizon = horizon
        self.task = task
        self.n_quantiles = n_quantiles
        self.dist = dist
        pools = [1, 2, 4][: max(n_blocks, 1)]
        while len(pools) < n_blocks:
            pools.append(pools[-1])
        self.blocks = nn.ModuleList(
            [NHitsBlock(lookback * n_features, horizon, hidden_size, pool_size=p) for p in pools]
        )
        out_dim = horizon
        if task == "quantile":
            out_dim = horizon * n_quantiles
        elif task == "distribution" or dist:
            out_dim = horizon * 2
        self.proj = nn.Linear(horizon, out_dim) if out_dim != horizon else nn.Identity()

    def forward(self, x: Any) -> Any:
        b = x.shape[0]
        residual = x.reshape(b, -1)
        forecast = 0.0
        for block in self.blocks:
            backcast, f = block(residual)
            residual = residual - backcast
            forecast = forecast + f
        out = self.proj(forecast)
        if self.task == "quantile":
            return out.view(b, self.horizon, self.n_quantiles)
        if self.task == "distribution" or self.dist:
            return out.view(b, self.horizon, 2)
        return out.view(b, self.horizon)
