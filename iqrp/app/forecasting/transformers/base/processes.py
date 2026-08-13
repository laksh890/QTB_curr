"""Synthetic long-range temporal datasets for transformer validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def simulate_long_range_series(
    n: int = 800,
    *,
    n_features: int = 6,
    n_assets: int = 1,
    noise: float = 0.1,
    regime: bool = True,
    classification: bool = False,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    gen = rng or np.random.default_rng(0)
    n_features = max(int(n_features), 2)
    t = np.arange(n, dtype=np.float64)
    X = gen.normal(size=(n, n_features))
    # long-range dependencies
    signal = 0.5 * np.sin(2 * np.pi * t / 48.0) + 0.3 * np.cos(2 * np.pi * t / 96.0)
    signal = signal + 0.4 * np.tanh(X[:, 0])
    if n_features > 2:
        signal = signal + 0.2 * X[:, 1] * X[:, 2]
    # delayed echo
    echo = np.roll(signal, 20)
    echo[:20] = signal[:20]
    signal = signal + 0.25 * echo
    regimes = ((t // 100) % 3).astype(int) if regime else np.zeros(n, dtype=int)
    signal = signal + 0.3 * regimes * X[:, min(1, n_features - 1)]
    y = signal + noise * gen.normal(size=n)
    if classification:
        y = (y > np.median(y)).astype(np.float64)
    data: dict[str, Any] = {"open_time": list(range(n)), "target": y}
    for j in range(n_features):
        data[f"f{j}"] = X[:, j]
    if regime:
        data["regime"] = regimes
    data["vol_forecast"] = np.abs(signal) * 0.08 + 0.01
    data["neural_forecast"] = 0.5 * signal + 0.05 * gen.normal(size=n)
    data["tree_forecast"] = 0.4 * signal + 0.05 * gen.normal(size=n)
    data["stat_forecast"] = np.roll(y, 1)
    data["stat_forecast"][0] = y[0]
    if n_assets > 1:
        data["asset_id"] = (t % n_assets).astype(int)
    return pl.DataFrame(data)


def feature_names(n_features: int = 6) -> list[str]:
    return [f"f{j}" for j in range(n_features)]
