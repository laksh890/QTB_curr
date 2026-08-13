"""Encoder-Decoder Seq2Seq with optional attention."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch
from iqrp.app.forecasting.neural.seq2seq.attention import Attention
from iqrp.app.forecasting.neural.seq2seq.decoder import Decoder
from iqrp.app.forecasting.neural.seq2seq.encoder import Encoder

try:
    import torch
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


class Seq2SeqNet(nn.Module if has_torch() else object):  # type: ignore[misc]
    def __init__(
        self,
        n_features: int,
        horizon: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_attention: bool = True,
        task: str = "regression",
        n_quantiles: int = 3,
        dist: bool = False,
    ) -> None:
        if has_torch():
            super().__init__()
        self.horizon = horizon
        self.use_attention = use_attention
        self.task = task
        self.n_quantiles = n_quantiles
        self.dist = dist
        self.encoder = Encoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, num_layers, dropout)
        self.attn = Attention(hidden_size) if use_attention else None
        self.combine = nn.Linear(hidden_size * 2, hidden_size) if use_attention else nn.Identity()
        out_dim = 1
        if task == "quantile":
            out_dim = n_quantiles
        elif task == "distribution" or dist:
            out_dim = 2
        self.out = nn.Linear(hidden_size, out_dim)

    def forward(self, x: Any) -> Any:
        b = x.shape[0]
        enc_out, state = self.encoder(x)
        # start token = last observed feature mean as proxy target
        y_prev = x[:, -1:, 0:1]
        outputs = []
        attn_maps = []
        h_t = state[0][-1]  # (B,H)
        for _ in range(self.horizon):
            if self.attn is not None:
                context, weights = self.attn(h_t, enc_out)
                attn_maps.append(weights)
                # feed decoder
                dec_in = y_prev
                dec_out, state = self.decoder(dec_in, state)
                h_t = state[0][-1]
                fused = torch.tanh(self.combine(torch.cat([h_t, context], dim=-1)))
                step = self.out(fused)
            else:
                dec_out, state = self.decoder(y_prev, state)
                h_t = state[0][-1]
                step = self.out(h_t)
            outputs.append(step)
            # teacher-free: next input is predicted mean
            if step.dim() == 2 and step.shape[-1] > 1:
                y_prev = step[:, :1].unsqueeze(1)
            else:
                y_prev = step.view(b, 1, 1)
        out = torch.stack(outputs, dim=1)  # (B,H,C) or (B,H,1)
        if self.task == "quantile":
            return out
        if self.task == "distribution" or self.dist:
            return out
        result = out.squeeze(-1)
        if attn_maps:
            # stash last attention on module for visualization
            self.last_attention = torch.stack(attn_maps, dim=1).detach()
        return result
