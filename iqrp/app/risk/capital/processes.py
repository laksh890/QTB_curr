"""Synthetic multi-strategy portfolios for capital-allocation validation.

May use ``iqrp.app.risk.processes`` for base market scenarios.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.risk.processes import simulate_risk_scenario

CapitalScenario = Literal[
    "independent",
    "correlated",
    "low_liquidity",
    "high_volatility",
    "regime",
    "drawdown",
    "tail",
]


def simulate_capital_scenario(
    kind: CapitalScenario = "independent",
    *,
    n: int = 400,
    n_strategies: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate multi-strategy returns + liquidity fields for capital tests."""
    rng = np.random.default_rng(seed)
    ns = max(int(n_strategies), 1)
    t = max(int(n), 20)
    names = [f"strat_{i}" for i in range(ns)]

    if kind == "independent":
        rets = rng.normal(0.0, 0.01, size=(t, ns))
        adv = np.full(ns, 1e7)
        spreads = np.full(ns, 0.0005)
        truth = {"kind": kind}
    elif kind == "correlated":
        base = simulate_risk_scenario("correlation_spike", n=t, n_assets=ns, seed=seed)
        rets = np.asarray(base["returns"], dtype=np.float64)
        adv = np.asarray(base["adv"], dtype=np.float64)
        spreads = np.asarray(base["spread"], dtype=np.float64)
        truth = {"kind": kind, "source": "risk.processes.correlation_spike"}
    elif kind == "low_liquidity":
        base = simulate_risk_scenario("low_liquidity", n=t, n_assets=ns, seed=seed)
        rets = np.asarray(base["returns"], dtype=np.float64)
        adv = np.asarray(base["adv"], dtype=np.float64)
        spreads = np.asarray(base["spread"], dtype=np.float64)
        truth = {"kind": kind, "source": "risk.processes.low_liquidity"}
    elif kind == "high_volatility":
        base = simulate_risk_scenario("high_volatility", n=t, n_assets=ns, seed=seed)
        rets = np.asarray(base["returns"], dtype=np.float64)
        adv = np.asarray(base["adv"], dtype=np.float64)
        spreads = np.asarray(base["spread"], dtype=np.float64)
        truth = {"kind": kind, "source": "risk.processes.high_volatility"}
    elif kind == "regime":
        base = simulate_risk_scenario("regime_transition", n=t, n_assets=ns, seed=seed)
        rets = np.asarray(base["returns"], dtype=np.float64)
        adv = np.asarray(base["adv"], dtype=np.float64)
        spreads = np.asarray(base["spread"], dtype=np.float64)
        truth = {"kind": kind, "source": "risk.processes.regime_transition"}
    elif kind == "drawdown":
        base = simulate_risk_scenario("drawdown", n=t, n_assets=ns, seed=seed)
        rets = np.asarray(base["returns"], dtype=np.float64)
        adv = np.asarray(base["adv"], dtype=np.float64)
        spreads = np.asarray(base["spread"], dtype=np.float64)
        truth = {"kind": kind, "source": "risk.processes.drawdown"}
    elif kind == "tail":
        # Fat tails via Student-t like mixture + correlated crashes
        corr = 0.35 * np.ones((ns, ns)) + 0.65 * np.eye(ns)
        cov = corr * (0.012**2)
        rets = rng.multivariate_normal(np.zeros(ns), cov, size=t)
        # Inject joint crash days
        crash = rng.choice(t, size=max(3, t // 50), replace=False)
        rets[crash] -= rng.uniform(0.04, 0.10, size=(crash.size, ns))
        adv = np.full(ns, 2e6)
        spreads = np.full(ns, 0.003)
        truth = {"kind": kind, "crash_days": crash.tolist()}
    else:
        rets = rng.normal(0.0, 0.01, size=(t, ns))
        adv = np.full(ns, 1e7)
        spreads = np.full(ns, 0.0005)
        truth = {"kind": str(kind)}

    vols = np.std(rets, axis=0, ddof=1)
    cov = np.cov(rets, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)

    return {
        "names": names,
        "returns": rets,
        "cov": cov,
        "vols": vols,
        "adv": adv,
        "spreads": spreads,
        "drawdowns": np.zeros(ns),
        "expected_opportunity": np.full(ns, 1.0 / ns),
        "forecast_confidence": np.full(ns, 0.7),
        "model_agreement": np.full(ns, 0.7),
        "truth": {**truth, "n": t, "n_strategies": ns, "seed": seed},
    }


def all_capital_scenarios(
    *,
    n: int = 300,
    n_strategies: int = 4,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Generate the full suite of capital validation scenarios."""
    kinds: list[CapitalScenario] = [
        "independent",
        "correlated",
        "low_liquidity",
        "high_volatility",
        "regime",
        "drawdown",
        "tail",
    ]
    return {
        k: simulate_capital_scenario(k, n=n, n_strategies=n_strategies, seed=seed + i)
        for i, k in enumerate(kinds)
    }
