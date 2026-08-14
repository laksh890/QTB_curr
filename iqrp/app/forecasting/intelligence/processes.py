"""Synthetic market frames for Forecast Intelligence validation.

Uses the Institutional Market Simulation Engine when available; falls back
to local generators so unit tests stay fast and dependency-light.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

MarketKind = Literal[
    "trending",
    "mean_reverting",
    "volatile",
    "regime_switching",
    "cross_asset",
]


def feature_names(n_features: int = 4) -> list[str]:
    return [f"f{i}" for i in range(n_features)]


def simulate_market_frame(
    n: int = 200,
    *,
    kind: MarketKind = "trending",
    n_features: int = 4,
    noise: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Build a supervised forecasting frame for intelligence benchmarks."""
    gen = rng or np.random.default_rng(0)
    try:
        return _from_simulation_engine(n, kind=kind, n_features=n_features, noise=noise, rng=gen)
    except Exception:
        return _local_simulate(n, kind=kind, n_features=n_features, noise=noise, rng=gen)


def _from_simulation_engine(
    n: int,
    *,
    kind: MarketKind,
    n_features: int,
    noise: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    from iqrp.app.simulation.base.simulator import MarketSimulator

    preset_map = {
        "trending": "bull",
        "mean_reverting": "sideways",
        "volatile": "high_volatility",
        "regime_switching": "mixed",
        "cross_asset": "mixed",
    }
    sim = MarketSimulator()
    market = sim.simulate_preset(preset_map[kind], n_steps=max(n + 5, 50))
    ohlcv = market.ohlcv()
    close = ohlcv["close"].to_numpy().astype(np.float64)
    rets = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=np.log(close[0]))
    if rets.size > n:
        rets = rets[-n:]
        close = close[-n:]
    gt = market.ground_truth
    regime = np.asarray(gt.regime_ids).reshape(-1)[-rets.size :]
    vol = np.asarray(gt.volatility).reshape(-1)[-rets.size :]
    return _assemble(
        rets, close, regime, vol, n_features=n_features, noise=noise, rng=rng, kind=kind
    )


def _local_simulate(
    n: int,
    *,
    kind: MarketKind,
    n_features: int,
    noise: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    t = np.arange(n, dtype=np.float64)
    if kind == "trending":
        rets = 0.001 + 0.01 * rng.normal(size=n)
        regime = np.zeros(n, dtype=np.int64)
    elif kind == "mean_reverting":
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + noise * rng.normal()
        rets = x
        regime = np.ones(n, dtype=np.int64)
    elif kind == "volatile":
        rets = 0.03 * rng.normal(size=n) * (1.0 + 0.5 * np.sin(t / 10))
        regime = np.full(n, 2, dtype=np.int64)
    elif kind == "regime_switching":
        regime = (t // max(n // 4, 1)).astype(np.int64) % 3
        drifts = np.array([0.002, -0.001, 0.0])
        vols = np.array([0.01, 0.02, 0.015])
        rets = drifts[regime] + vols[regime] * rng.normal(size=n)
    else:  # cross_asset
        rets = 0.001 + 0.015 * rng.normal(size=n)
        regime = (rng.random(n) > 0.5).astype(np.int64)
    close = 100.0 * np.exp(np.cumsum(rets))
    vol = np.abs(rets) * 10
    return _assemble(
        rets, close, regime, vol, n_features=n_features, noise=noise, rng=rng, kind=kind
    )


def _assemble(
    rets: np.ndarray,
    close: np.ndarray,
    regime: np.ndarray,
    vol: np.ndarray,
    *,
    n_features: int,
    noise: float,
    rng: np.random.Generator,
    kind: str,
) -> pl.DataFrame:
    n = rets.size
    feats = {}
    for i in range(n_features):
        lag = min(i + 1, n - 1)
        base = np.roll(rets, lag)
        base[:lag] = 0.0
        feats[f"f{i}"] = base + noise * rng.normal(size=n)
    # target: next-step return
    target = np.roll(rets, -1)
    target[-1] = target[-2] if n > 1 else 0.0
    data = {
        "open_time": np.arange(n),
        **feats,
        "target": target,
        "close": close,
        "regime": regime.astype(np.int64),
        "vol_forecast": vol.astype(np.float64),
        "asset_id": ["A"] * n if kind != "cross_asset" else (["A", "B"] * ((n + 1) // 2))[:n],
        "spread": (0.0001 + 0.0005 * rng.random(n)).tolist(),
    }
    return pl.DataFrame(data)
