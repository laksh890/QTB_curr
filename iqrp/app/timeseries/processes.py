"""Synthetic processes with known structure for analytical recovery tests.

Uses the Institutional Market Simulation Engine when available; otherwise
local generators with known ground-truth properties.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

ProcessKind = Literal[
    "stationary",
    "non_stationary",
    "mean_reverting",
    "trending",
    "periodic",
    "long_memory",
    "structural_break",
    "regime_change",
    "cointegrated",
    "anomalous",
]


def simulate_process(
    kind: ProcessKind,
    n: int = 256,
    *,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return series (+ optional companion) and ground-truth metadata."""
    rng = np.random.default_rng(seed)
    if kind == "stationary":
        x = rng.normal(0, 1, size=n)
        return {"series": x, "truth": {"stationary": True, "kind": kind}}
    if kind == "non_stationary":
        x = np.cumsum(rng.normal(0, 1, size=n))
        return {"series": x, "truth": {"stationary": False, "kind": kind}}
    if kind == "mean_reverting":
        x = np.zeros(n)
        phi = float(kwargs.get("phi", 0.7))
        for i in range(1, n):
            x[i] = phi * x[i - 1] + 0.5 * rng.normal()
        return {"series": x, "truth": {"mean_reverting": True, "phi": phi, "kind": kind}}
    if kind == "trending":
        t = np.arange(n, dtype=np.float64)
        x = 0.05 * t + rng.normal(0, 1, size=n)
        return {"series": x, "truth": {"trending": True, "kind": kind}}
    if kind == "periodic":
        period = int(kwargs.get("period", 24))
        t = np.arange(n, dtype=np.float64)
        x = np.sin(2 * np.pi * t / period) + 0.2 * rng.normal(size=n)
        return {"series": x, "truth": {"period": period, "kind": kind}}
    if kind == "long_memory":
        # approximate via fractional integration of noise (simple ARFIMA-lite)
        e = rng.normal(size=n)
        d = float(kwargs.get("d", 0.3))
        x = np.zeros(n)
        w = np.array([1.0])
        for i in range(n):
            x[i] = np.dot(w[::-1], e[: i + 1][-len(w) :]) if i else e[0]
            # update binomial weights for (1-L)^-d
            k = i + 1
            w = np.append(w, w[-1] * (k + d - 1) / k)
            if w.size > 40:
                w = w[-40:]
        return {"series": x, "truth": {"long_memory": True, "d": d, "kind": kind}}
    if kind == "structural_break":
        brk = int(kwargs.get("break_at", n // 2))
        x = np.concatenate([rng.normal(0, 1, brk), rng.normal(3, 1, n - brk)])
        return {"series": x, "truth": {"break_at": brk, "kind": kind}}
    if kind == "regime_change":
        mid = n // 2
        x = np.concatenate([rng.normal(0, 0.5, mid), rng.normal(0, 2.0, n - mid)])
        return {"series": x, "truth": {"regime_boundary": mid, "kind": kind}}
    if kind == "cointegrated":
        e = rng.normal(size=n)
        common = np.cumsum(e)
        x = common + rng.normal(0, 0.2, size=n)
        y = 2.0 * common + rng.normal(0, 0.2, size=n)
        return {
            "series": x,
            "series_y": y,
            "truth": {"cointegrated": True, "beta": 2.0, "kind": kind},
        }
    if kind == "anomalous":
        x = rng.normal(0, 1, size=n)
        idx = sorted(rng.choice(n, size=max(3, n // 50), replace=False).tolist())
        x[idx] = x[idx] + 8.0
        return {"series": x, "truth": {"anomaly_indices": idx, "kind": kind}}
    # fallback
    x = rng.normal(size=n)
    return {"series": x, "truth": {"kind": kind}}


def from_market_simulator(
    n: int = 200, *, preset: str = "sideways", seed: int = 0
) -> dict[str, Any]:
    try:
        from iqrp.app.simulation.base.simulator import MarketSimulator

        sim = MarketSimulator()
        market = sim.simulate_preset(preset, n_steps=max(n + 5, 50))
        close = market.ohlcv()["close"].to_numpy().astype(np.float64)
        if close.size > n:
            close = close[-n:]
        return {
            "series": close,
            "truth": {"source": "MarketSimulator", "preset": preset, "seed": seed},
        }
    except Exception as exc:
        rng = np.random.default_rng(seed)
        return {
            "series": np.cumsum(rng.normal(0, 0.01, size=n)) + 100,
            "truth": {"source": "fallback", "error": str(exc)},
        }
