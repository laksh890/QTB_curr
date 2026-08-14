"""DeepAR-style probabilistic RNN forecaster."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class DeepARNet(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        horizon: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        distribution: str = "gaussian",
    ) -> None:
        if has_torch():
            super().__init__()
        self.horizon = horizon
        self.distribution = distribution
        self.encoder = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mu = nn.Linear(hidden_size, horizon)
        self.sigma = nn.Linear(hidden_size, horizon)
        self.df = nn.Linear(hidden_size, 1) if distribution == "student_t" else None

    def forward(self, x: Any) -> Any:
        out, _ = self.encoder(x)
        h = out[:, -1, :]
        mu = self.mu(h)
        log_sigma = self.sigma(h)
        # pack as (B, H, 2)
        return torch.stack([mu, log_sigma], dim=-1)
