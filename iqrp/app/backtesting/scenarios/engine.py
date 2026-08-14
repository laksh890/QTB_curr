"""Unified backtest scenario engine."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.backtesting.scenarios.correlation import run_correlation_scenario
from iqrp.app.backtesting.scenarios.gap import run_gap_scenario
from iqrp.app.backtesting.scenarios.historical import (
    HistoricalScenario,
    run_historical_scenario,
)
from iqrp.app.backtesting.scenarios.hypothetical import (
    HypotheticalShock,
    run_hypothetical_scenario,
)
from iqrp.app.backtesting.scenarios.liquidity import run_liquidity_scenario
from iqrp.app.backtesting.scenarios.monte_carlo import run_monte_carlo
from iqrp.app.backtesting.scenarios.regime import (
    classify_simple_regimes,
    run_regime_scenario,
)
from iqrp.app.backtesting.scenarios.volatility import run_volatility_scenario

ScenarioKind = Literal[
    "historical",
    "hypothetical",
    "monte_carlo",
    "regime",
    "liquidity",
    "volatility",
    "correlation",
    "gap",
]

__all__ = ["ScenarioEngine", "ScenarioKind"]


class ScenarioEngine:
    """Orchestrate historical, hypothetical, Monte Carlo, and regime scenarios.

    Path-dependent state (positions, costs, drawdown) can be threaded via
    ``state`` and is echoed in results for downstream simulators.
    """

    def __init__(
        self,
        *,
        n_simulations: int = 500,
        seed: int = 42,
        periods_per_year: float = 252.0,
        block_size: int = 5,
    ) -> None:
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)
        self.periods_per_year = float(periods_per_year)
        self.block_size = int(block_size)

    def run(
        self,
        kind: ScenarioKind | str,
        returns: Any,
        *,
        scenario: HistoricalScenario | dict[str, Any] | None = None,
        shocks: list[HypotheticalShock | dict[str, Any]] | None = None,
        method: str = "bootstrap",
        regimes: Any | None = None,
        weights: Any | None = None,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dispatch to the selected scenario runner.

        ``state`` may include position/portfolio/drawdown/cost/risk keys that
        are preserved (path-dependent context) on the result under ``state``.
        """
        k = str(kind).lower()
        if k == "historical":
            if scenario is None:
                raise ValueError(
                    "historical scenarios require a user-defined scenario "
                    "(start/end/mask) — no hard-coded crises"
                )
            result = run_historical_scenario(
                returns,
                scenario,
                weights=weights,
                periods_per_year=self.periods_per_year,
            )
        elif k == "hypothetical":
            if not shocks:
                raise ValueError("hypothetical scenarios require shocks")
            result = run_hypothetical_scenario(returns, shocks, weights=weights, **kwargs)
        elif k in ("monte_carlo", "mc"):
            result = run_monte_carlo(
                returns,
                method=method,  # type: ignore[arg-type]
                n_simulations=kwargs.pop("n_simulations", self.n_simulations),
                seed=kwargs.pop("seed", self.seed),
                block_size=kwargs.pop("block_size", self.block_size),
                periods_per_year=self.periods_per_year,
                regimes=regimes,
                **kwargs,
            )
        elif k == "regime":
            labs = regimes if regimes is not None else classify_simple_regimes(returns)
            result = run_regime_scenario(
                returns,
                labs,
                regime=kwargs.pop("regime", None),
                periods_per_year=self.periods_per_year,
            )
            result["regimes"] = np.asarray(labs)
        elif k == "liquidity":
            result = run_liquidity_scenario(returns, **kwargs)
        elif k == "volatility":
            result = run_volatility_scenario(
                returns, periods_per_year=self.periods_per_year, **kwargs
            )
        elif k == "correlation":
            result = run_correlation_scenario(returns, seed=kwargs.pop("seed", self.seed), **kwargs)
        elif k == "gap":
            result = run_gap_scenario(returns, **kwargs)
        else:
            raise ValueError(f"unknown scenario kind: {kind!r}")

        if state is not None:
            result = dict(result)
            result["state"] = dict(state)
        return result

    def run_suite(
        self,
        returns: Any,
        *,
        historical: HistoricalScenario | dict[str, Any] | None = None,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a configurable suite of scenarios and collect reports."""
        kinds = include or ["historical", "volatility", "liquidity", "gap", "monte_carlo"]
        reports: dict[str, Any] = {}
        for kind in kinds:
            if kind == "historical" and historical is None:
                continue
            try:
                reports[kind] = self.run(
                    kind,  # type: ignore[arg-type]
                    returns,
                    scenario=historical,
                    **kwargs,
                )
            except Exception as exc:
                reports[kind] = {"error": str(exc), "kind": kind}
        return {"name": "scenario_suite", "reports": reports}
