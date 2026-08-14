"""Synthetic portfolio scenarios via risk.processes / simulation when available."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

ScenarioKind = Literal[
    "normal",
    "high_volatility",
    "low_liquidity",
    "correlation_spike",
    "regime_transition",
    "large_gaps",
    "drawdown",
]


def simulate_portfolio_scenario(
    kind: ScenarioKind | str = "normal",
    *,
    n: int = 500,
    n_assets: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate multi-asset returns for portfolio construction stress tests.

    Prefers ``iqrp.app.risk.processes.simulate_risk_scenario``; falls back to a
    local Gaussian generator when risk processes are unavailable.
    """
    try:
        from iqrp.app.risk.processes import simulate_risk_scenario

        out = simulate_risk_scenario(
            kind=kind,  # type: ignore[arg-type]
            n=n,
            n_assets=n_assets,
            seed=seed,
        )
        out = dict(out)
        out.setdefault("source", "iqrp.app.risk.processes")
        return out
    except Exception as exc:
        rng = np.random.default_rng(int(seed))
        na = max(int(n_assets), 1)
        t = max(int(n), 10)
        vol = 0.01
        corr = 0.2
        if str(kind) == "high_volatility":
            vol = 0.04
            corr = 0.3
        elif str(kind) == "correlation_spike":
            corr = 0.85
            vol = 0.02
        elif str(kind) == "drawdown":
            vol = 0.015
        c = corr * np.ones((na, na)) + (1.0 - corr) * np.eye(na)
        cov = c * (vol**2)
        rets = rng.multivariate_normal(np.zeros(na), cov, size=t)
        if str(kind) == "drawdown":
            start = t // 3
            end = start + t // 5
            rets[start:end] -= 0.015
        return {
            "name": "simulate_portfolio_scenario",
            "kind": str(kind),
            "returns": rets,
            "adv": np.full(na, 1e6),
            "spread": np.full(na, 0.001),
            "n": t,
            "n_assets": na,
            "seed": int(seed),
            "source": "local_fallback",
            "fallback_reason": str(exc),
        }


def monte_carlo_portfolio_paths(
    returns: Any,
    *,
    n_simulations: int = 1000,
    horizon: int = 21,
    seed: int = 42,
    weights: Any | None = None,
) -> dict[str, Any]:
    """Simulate portfolio P&L paths via risk.simulation when available."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim == 1:
        R = R.reshape(-1, 1)
    t, n = R.shape
    w = np.asarray(
        weights if weights is not None else np.full(n, 1.0 / max(n, 1)), dtype=np.float64
    )
    if w.size != n:
        w = np.resize(w, n)
        s = float(np.sum(np.abs(w)))
        w = w / s if s > 1e-12 else np.full(n, 1.0 / n)

    try:
        from iqrp.app.risk.simulation import correlated_monte_carlo

        # correlated_monte_carlo expects returns calibration
        sim = correlated_monte_carlo(R, n_simulations=n_simulations, horizon=horizon, seed=seed)
        paths = sim.get("paths")
        # If paths are asset-level, collapse with weights
        if isinstance(paths, np.ndarray) and paths.ndim == 3:
            # (n_sim, horizon, n_assets)
            port = paths @ w
        elif isinstance(paths, np.ndarray) and paths.ndim == 2:
            port = paths
        else:
            port = None
        out = dict(sim)
        out["portfolio_paths"] = port
        out["weights"] = w.tolist()
        out["source"] = "iqrp.app.risk.simulation"
        return out
    except Exception as exc:
        rng = np.random.default_rng(int(seed))
        mu = R.mean(axis=0)
        cov = np.cov(R.T) if t > 1 else np.eye(n) * 1e-4
        cov = np.atleast_2d(cov)
        n_sim = max(int(n_simulations), 1)
        h = max(int(horizon), 1)
        asset_paths = rng.multivariate_normal(mu, cov + 1e-10 * np.eye(n), size=(n_sim, h))
        port = asset_paths @ w
        return {
            "name": "monte_carlo_portfolio_paths",
            "portfolio_paths": port,
            "terminal": port.sum(axis=1),
            "weights": w.tolist(),
            "n_simulations": n_sim,
            "horizon": h,
            "seed": int(seed),
            "source": "local_fallback",
            "fallback_reason": str(exc),
        }


def process_scenarios(
    kinds: list[str] | None = None,
    *,
    n: int = 250,
    n_assets: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Batch synthetic scenarios for portfolio construction validation."""
    default_kinds = [
        "normal",
        "high_volatility",
        "correlation_spike",
        "regime_transition",
        "drawdown",
    ]
    use = list(kinds) if kinds is not None else default_kinds
    scenarios = {
        k: simulate_portfolio_scenario(k, n=n, n_assets=n_assets, seed=seed + i)
        for i, k in enumerate(use)
    }
    return {
        "name": "process_scenarios",
        "kinds": use,
        "scenarios": scenarios,
        "n": n,
        "n_assets": n_assets,
        "seed": int(seed),
    }
