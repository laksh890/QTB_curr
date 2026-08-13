"""Unified scenario engine combining simulation methods."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns, as_weights
from iqrp.app.risk.simulation.bootstrap import block_bootstrap, historical_bootstrap
from iqrp.app.risk.simulation.copula import gaussian_copula_simulate
from iqrp.app.risk.simulation.monte_carlo import correlated_monte_carlo, parametric_monte_carlo

Method = Literal[
    "parametric",
    "correlated",
    "bootstrap",
    "block_bootstrap",
    "gaussian_copula",
]


class ScenarioEngine:
    """Run risk simulations and summarize terminal P&L distribution."""

    def __init__(
        self,
        *,
        n_simulations: int = 5000,
        horizon: int = 1,
        seed: int = 42,
        block_size: int = 5,
    ) -> None:
        self.n_simulations = int(n_simulations)
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.block_size = int(block_size)

    def run(
        self,
        returns: Any,
        *,
        method: Method = "parametric",
        weights: Any | None = None,
        cov: Any | None = None,
        mean: Any | None = None,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """Execute the selected simulator and return distributional summary.

        For multivariate methods, portfolio terminal P&L uses ``weights``.
        """
        m = str(method).lower()
        r = np.asarray(returns, dtype=np.float64)

        if m == "correlated":
            if r.ndim == 2:
                mu = mean if mean is not None else np.mean(r, axis=0)
                c = cov if cov is not None else np.cov(r, rowvar=False)
            else:
                mu = mean if mean is not None else np.array([float(np.mean(as_returns(r)))])
                sig = float(np.std(as_returns(r), ddof=1)) if as_returns(r).size > 1 else 0.0
                c = cov if cov is not None else np.array([[sig ** 2]])
            sim = correlated_monte_carlo(
                mu, c, n_simulations=self.n_simulations, horizon=self.horizon, seed=self.seed
            )
            terminal_assets = sim["terminal"]
            w = as_weights(weights if weights is not None else 1.0, n=terminal_assets.shape[1])
            terminal = terminal_assets @ w
            detail = {k: v for k, v in sim.items() if k not in ("paths", "terminal")}
        elif m == "bootstrap":
            series = as_returns(r if r.ndim == 1 else (r @ as_weights(weights, n=r.shape[1])))
            sim = historical_bootstrap(
                series, n_simulations=self.n_simulations, horizon=self.horizon, seed=self.seed
            )
            terminal = sim["terminal"]
            detail = {k: v for k, v in sim.items() if k not in ("paths", "terminal")}
        elif m == "block_bootstrap":
            series = as_returns(r if r.ndim == 1 else (r @ as_weights(weights, n=r.shape[1])))
            sim = block_bootstrap(
                series,
                n_simulations=self.n_simulations,
                horizon=self.horizon,
                block_size=self.block_size,
                seed=self.seed,
            )
            terminal = sim["terminal"]
            detail = {k: v for k, v in sim.items() if k not in ("paths", "terminal")}
        elif m == "gaussian_copula":
            if r.ndim == 1:
                r2 = r.reshape(-1, 1)
            else:
                r2 = r
            sim = gaussian_copula_simulate(r2, n_simulations=self.n_simulations, seed=self.seed)
            samples = sim["samples"]
            w = as_weights(weights if weights is not None else 1.0, n=samples.shape[1])
            # Single-period copula draw treated as horizon return
            terminal = samples @ w
            if self.horizon > 1:
                # Scale via sqrt-time for summary only (independent increments assumption)
                terminal = terminal * np.sqrt(self.horizon)
            detail = {k: v for k, v in sim.items() if k != "samples"}
        else:
            series = as_returns(r if r.ndim == 1 else (r @ as_weights(weights, n=r.shape[1])))
            sim = parametric_monte_carlo(
                series, n_simulations=self.n_simulations, horizon=self.horizon, seed=self.seed
            )
            terminal = sim["terminal"]
            detail = {k: v for k, v in sim.items() if k not in ("paths", "terminal")}
            m = "parametric"

        conf = float(confidence)
        alpha = 1.0 - conf
        var = float(max(-np.quantile(terminal, alpha), 0.0)) if terminal.size else 0.0
        thr = float(np.quantile(terminal, alpha)) if terminal.size else 0.0
        tail = terminal[terminal <= thr] if terminal.size else np.array([])
        es = float(max(-np.mean(tail), 0.0)) if tail.size else var

        return {
            "name": "scenario_engine",
            "method": m,
            "n_simulations": int(self.n_simulations),
            "horizon": int(self.horizon),
            "seed": int(self.seed),
            "terminal_mean": float(np.mean(terminal)) if terminal.size else 0.0,
            "terminal_std": float(np.std(terminal, ddof=1)) if terminal.size > 1 else 0.0,
            "var": RiskMeasure(
                name="var",
                value=var,
                unit="return",
                confidence=conf,
                horizon=int(self.horizon),
                method=f"scenario_{m}",
            ).to_dict(),
            "expected_shortfall": RiskMeasure(
                name="expected_shortfall",
                value=es,
                unit="return",
                confidence=conf,
                horizon=int(self.horizon),
                method=f"scenario_{m}",
            ).to_dict(),
            "detail": detail,
        }
