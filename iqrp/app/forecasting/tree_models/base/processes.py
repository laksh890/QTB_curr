"""Synthetic nonlinear financial datasets for simulation validation."""

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
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    gen = rng or np.random.default_rng(0)
    n_features = max(int(n_features), 2)
    X = gen.normal(size=(n, n_features))
    # nonlinear signal (safe for small feature counts)
    signal = 0.8 * np.tanh(X[:, 0])
    if n_features > 2:
        signal = signal + 0.5 * X[:, 1] * X[:, 2]
    if n_features > 3:
        signal = signal + -0.3 * np.sin(X[:, 3])
    if n_features > 4:
        signal = signal + 0.2 * X[:, 4] ** 2
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
    # optional vol / validated feature proxies
    data["vol_forecast"] = np.abs(signal) * 0.1 + 0.01
    return pl.DataFrame(data)


def feature_names(n_features: int = 6) -> list[str]:
    return [f"f{j}" for j in range(n_features)]
