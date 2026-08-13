"""Output heads for transformer forecasting."""

from __future__ import annotations

from typing import Any

from iqrp.app.forecasting.neural.base.torch_utils import has_torch

try:
    from torch import nn
except Exception:  # noqa: BLE001  # pragma: no cover
    nn = object  # type: ignore[assignment]


def forecast_head(
    d_model: int,
    horizon: int,
    *,
    task: str = "regression",
    n_classes: int = 2,
    n_quantiles: int = 3,
    n_mixtures: int = 3,
    dist: bool = False,
) -> Any:
    if task in {"classification", "multiclass"}:
        return nn.Linear(d_model, horizon * n_classes)
    if task in {"binary", "probability"}:
        return nn.Linear(d_model, horizon)
    if task == "quantile":
        return nn.Linear(d_model, horizon * n_quantiles)
    if task == "distribution" or dist:
        return nn.Linear(d_model, horizon * 2)
    if task == "mixture":
        # pi, mu, log_sigma per component per horizon — simplified flat
        return nn.Linear(d_model, horizon * n_mixtures * 3)
    return nn.Linear(d_model, horizon)


def reshape_forecast(
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
