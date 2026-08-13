"""N-BEATS backbone (generic + interpretable trend/seasonality blocks)."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class NBeatsBlock(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(self, lookback: int, horizon: int, hidden: int, n_layers: int = 2) -> None:
        if has_torch():
            super().__init__()
        layers = []
        in_dim = lookback
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(hidden, lookback)
        self.theta_f = nn.Linear(hidden, horizon)

    def forward(self, x: Any) -> tuple[Any, Any]:
        h = self.fc(x)
        backcast = self.theta_b(h)
        forecast = self.theta_f(h)
        return backcast, forecast


class NBeatsNet(nn.Module if has_torch() else object):  # type: ignore[misc]
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
        self.lookback = lookback
        self.horizon = horizon
        self.task = task
        self.n_quantiles = n_quantiles
        self.dist = dist
        self.n_features = n_features
        self.blocks = nn.ModuleList(
            [NBeatsBlock(lookback * n_features, horizon, hidden_size) for _ in range(max(n_blocks, 1))]
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
