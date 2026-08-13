"""Synthetic nonlinear financial datasets for neural forecast validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def simulate_nonlinear_returns(
    n: int = 400,
    *,
    n_features: int = 6,
    noise: float = 0.1,
    regime: bool = True,
    classification: bool = False,
    seasonal: bool = True,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    gen = rng or np.random.default_rng(0)
    n_features = max(int(n_features), 2)
    t = np.arange(n, dtype=np.float64)
    X = gen.normal(size=(n, n_features))
    signal = 0.8 * np.tanh(X[:, 0])
    if n_features > 2:
        signal = signal + 0.5 * X[:, 1] * X[:, 2]
    if n_features > 3:
        signal = signal + -0.3 * np.sin(X[:, 3])
    if n_features > 4:
        signal = signal + 0.2 * X[:, 4] ** 2
    if seasonal:
        signal = signal + 0.15 * np.sin(2 * np.pi * t / 20.0) + 0.1 * np.cos(2 * np.pi * t / 40.0)
    regimes = (X[:, 0] > 0).astype(int) if regime else np.zeros(n, dtype=int)
    signal = signal + 0.4 * regimes * X[:, min(1, n_features - 1)]
    y = signal + noise * gen.normal(size=n)
    if classification:
        y = (y > np.median(y)).astype(np.float64)
    data: dict[str, Any] = {"open_time": list(range(n)), "target": y}
    for j in range(n_features):
        data[f"f{j}"] = X[:, j]
    if regime:
        data["regime"] = regimes
    data["vol_forecast"] = np.abs(signal) * 0.1 + 0.01
    data["stat_forecast"] = np.roll(y, 1)
    data["stat_forecast"][0] = y[0]
    data["tree_forecast"] = 0.5 * signal + 0.05 * gen.normal(size=n)
    return pl.DataFrame(data)


def feature_names(n_features: int = 6) -> list[str]:
    return [f"f{j}" for j in range(n_features)]


def multi_horizon_frame(n: int = 300, horizon: int = 5, rng: np.random.Generator | None = None) -> pl.DataFrame:
    """Frame with explicit multi-step target columns for multi-target tests."""
    frame = simulate_nonlinear_returns(n, n_features=4, rng=rng)
    y = frame["target"].to_numpy()
    for h in range(1, horizon + 1):
        col = np.roll(y, -h)
        col[-h:] = y[-1]
        frame = frame.with_columns(pl.Series(f"target_h{h}", col))
    return frame
