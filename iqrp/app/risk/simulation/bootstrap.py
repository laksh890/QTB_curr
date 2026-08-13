"""Historical and block bootstrap resampling."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import as_returns


def historical_bootstrap(
    returns: Any,
    *,
    n_simulations: int = 5000,
    horizon: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """i.i.d. bootstrap of historical returns (with replacement)."""
    r = as_returns(returns)
    n_sim = max(int(n_simulations), 1)
    h = max(int(horizon), 1)
    rng = np.random.default_rng(int(seed))
    if r.size == 0:
        paths = np.zeros((n_sim, h), dtype=np.float64)
    else:
        idx = rng.integers(0, r.size, size=(n_sim, h))
        paths = r[idx]
    terminal = paths.sum(axis=1)
    return {
        "name": "historical_bootstrap",
        "paths": paths,
        "terminal": terminal,
        "n_obs": int(r.size),
        "n_simulations": n_sim,
        "horizon": h,
        "seed": int(seed),
    }


def block_bootstrap(
    returns: Any,
    *,
    n_simulations: int = 5000,
    horizon: int | None = None,
    block_size: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Moving-block bootstrap preserving short-run dependence.

    Each path is built by concatenating random contiguous blocks until
    length >= horizon (default: full sample length).
    """
    r = as_returns(returns)
    n = r.size
    n_sim = max(int(n_simulations), 1)
    bs = max(int(block_size), 1)
    h = int(horizon) if horizon is not None else max(n, 1)
    h = max(h, 1)
    rng = np.random.default_rng(int(seed))

    if n == 0:
        paths = np.zeros((n_sim, h), dtype=np.float64)
    else:
        max_start = max(n - bs, 0)
        paths = np.empty((n_sim, h), dtype=np.float64)
        for i in range(n_sim):
            collected: list[float] = []
            while len(collected) < h:
                start = int(rng.integers(0, max_start + 1)) if n > bs else 0
                block = r[start : start + bs]
                if block.size == 0:
                    block = r
                collected.extend(block.tolist())
            paths[i] = np.asarray(collected[:h], dtype=np.float64)

    terminal = paths.sum(axis=1)
    return {
        "name": "block_bootstrap",
        "paths": paths,
        "terminal": terminal,
        "n_obs": n,
        "n_simulations": n_sim,
        "horizon": h,
        "block_size": bs,
        "seed": int(seed),
    }
