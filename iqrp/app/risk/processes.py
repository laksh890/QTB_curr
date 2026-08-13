"""Synthetic market scenarios for risk validation (simulation-engine aware).

Generates portfolios under normal / high-vol / low-liquidity / correlation-spike /
regime-transition / gap / drawdown regimes without modifying historical data.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

RiskScenario = Literal[
    "normal",
    "high_volatility",
    "low_liquidity",
    "correlation_spike",
    "regime_transition",
    "large_gaps",
    "drawdown",
]


def simulate_risk_scenario(
    kind: RiskScenario = "normal",
    *,
    n: int = 500,
    n_assets: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Return multi-asset returns (+ optional liquidity fields) and truth metadata."""
    rng = np.random.default_rng(seed)
    na = max(int(n_assets), 1)
    t = max(int(n), 10)

    if kind == "normal":
        corr = 0.2 * np.ones((na, na)) + 0.8 * np.eye(na)
        cov = corr * (0.01**2)
        rets = rng.multivariate_normal(np.zeros(na), cov, size=t)
        adv = np.full(na, 1e7)
        spread = np.full(na, 0.0005)
    elif kind == "high_volatility":
        corr = 0.3 * np.ones((na, na)) + 0.7 * np.eye(na)
        cov = corr * (0.04**2)
        rets = rng.multivariate_normal(np.zeros(na), cov, size=t)
        adv = np.full(na, 5e6)
        spread = np.full(na, 0.001)
    elif kind == "low_liquidity":
        corr = 0.15 * np.ones((na, na)) + 0.85 * np.eye(na)
        cov = corr * (0.015**2)
        rets = rng.multivariate_normal(np.zeros(na), cov, size=t)
        adv = np.full(na, 5e4)
        spread = np.full(na, 0.01)
    elif kind == "correlation_spike":
        mid = t // 2
        low = 0.1 * np.ones((na, na)) + 0.9 * np.eye(na)
        high = 0.9 * np.ones((na, na)) + 0.1 * np.eye(na)
        r1 = rng.multivariate_normal(np.zeros(na), low * (0.01**2), size=mid)
        r2 = rng.multivariate_normal(np.zeros(na), high * (0.02**2), size=t - mid)
        rets = np.vstack([r1, r2])
        adv = np.full(na, 2e6)
        spread = np.full(na, 0.002)
    elif kind == "regime_transition":
        mid = t // 2
        r1 = rng.normal(0.0005, 0.008, size=(mid, na))
        r2 = rng.normal(-0.001, 0.03, size=(t - mid, na))
        rets = np.vstack([r1, r2])
        adv = np.full(na, 3e6)
        spread = np.full(na, 0.0015)
    elif kind == "large_gaps":
        rets = rng.normal(0, 0.01, size=(t, na))
        gap_idx = rng.choice(t, size=max(3, t // 40), replace=False)
        rets[gap_idx] -= rng.uniform(0.05, 0.12, size=(gap_idx.size, na))
        adv = np.full(na, 1e6)
        spread = np.full(na, 0.003)
    elif kind == "drawdown":
        rets = rng.normal(0.0002, 0.01, size=(t, na))
        start = t // 3
        end = start + t // 5
        rets[start:end] -= 0.015
        adv = np.full(na, 4e6)
        spread = np.full(na, 0.001)
    else:
        rets = rng.normal(0, 0.01, size=(t, na))
        adv = np.full(na, 1e7)
        spread = np.full(na, 0.0005)

    weights = np.full(na, 1.0 / na)
    return {
        "returns": rets,
        "weights": weights,
        "adv": adv,
        "spread": spread,
        "truth": {"kind": kind, "n": t, "n_assets": na, "seed": seed},
    }


def from_market_simulator(
    n: int = 200,
    *,
    preset: str = "sideways",
    seed: int = 0,
) -> dict[str, Any]:
    """Pull returns from Institutional Market Simulation Engine when available."""
    try:
        from iqrp.app.simulation.base.simulator import MarketSimulator

        sim = MarketSimulator()
        market = sim.simulate_preset(preset, n_steps=max(n + 5, 50))
        prices = np.asarray(getattr(market, "prices", None) or market.get("prices"), dtype=np.float64)
        if prices.ndim == 1:
            rets = np.diff(np.log(np.maximum(prices, 1e-12)))
            rets = rets.reshape(-1, 1)
        else:
            rets = np.diff(np.log(np.maximum(prices, 1e-12)), axis=0)
        rets = rets[-n:]
        na = rets.shape[1]
        return {
            "returns": rets,
            "weights": np.full(na, 1.0 / na),
            "adv": np.full(na, 1e7),
            "spread": np.full(na, 0.0005),
            "truth": {"kind": "market_simulator", "preset": preset, "seed": seed},
            "source": "iqrp.app.simulation",
        }
    except Exception as exc:  # noqa: BLE001
        out = simulate_risk_scenario("normal", n=n, n_assets=1, seed=seed)
        out["truth"]["fallback"] = str(exc)
        out["source"] = "local_fallback"
        return out
