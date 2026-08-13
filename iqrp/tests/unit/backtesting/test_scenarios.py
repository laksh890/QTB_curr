"""Scenario testing: historical, hypothetical, Monte Carlo, regime, etc."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.backtesting.scenarios import (
    HistoricalScenario,
    ScenarioEngine,
    run_historical_scenario,
    run_hypothetical_scenario,
    run_monte_carlo,
)
from iqrp.app.backtesting.scenarios.correlation import (
    apply_correlation_shock,
    run_correlation_scenario,
    stress_correlation,
)
from iqrp.app.backtesting.scenarios.gap import apply_gap_shock, run_gap_scenario
from iqrp.app.backtesting.scenarios.historical import slice_window
from iqrp.app.backtesting.scenarios.hypothetical import HypotheticalShock, apply_hypothetical_shock
from iqrp.app.backtesting.scenarios.liquidity import apply_liquidity_shock, run_liquidity_scenario
from iqrp.app.backtesting.scenarios.monte_carlo import (
    block_bootstrap_paths,
    bootstrap_paths,
    correlated_paths,
    regime_conditioned_paths,
    residual_bootstrap_paths,
    trade_bootstrap_paths,
)
from iqrp.app.backtesting.scenarios.regime import (
    classify_simple_regimes,
    evaluate_regime_robustness,
    run_regime_scenario,
)
from iqrp.app.backtesting.scenarios.volatility import apply_volatility_shock, run_volatility_scenario


def test_historical_user_defined_no_hardcoded_crises(short_returns) -> None:
    sc = HistoricalScenario(name="user_window", start=10, end=40)
    out = run_historical_scenario(short_returns, sc)
    assert out["kind"] == "historical"
    assert out["n_obs"] == 30
    assert out["name"] == "user_window"

    # dict form + mask
    mask = np.zeros(len(short_returns), dtype=bool)
    mask[5:15] = True
    out2 = run_historical_scenario(
        short_returns, {"name": "masked", "mask": mask, "market_conditions": {"vol": "high"}}
    )
    assert out2["n_obs"] == 10

    # multi-asset
    assets = np.column_stack([short_returns, short_returns * 0.8])
    out3 = run_historical_scenario(assets, HistoricalScenario("m", start=0, end=20), weights=[0.5, 0.5])
    assert out3["n_obs"] == 20

    w = slice_window(short_returns, start=0, end=10)
    assert w.size == 10
    with pytest.raises(ValueError):
        slice_window(short_returns, start=20, end=10)


def test_hypothetical_shocks(short_returns) -> None:
    kinds = [
        "price",
        "volatility",
        "correlation",
        "liquidity",
        "spread",
        "cost",
        "interest_rate",
        "fx",
        "gap",
    ]
    shocks = [HypotheticalShock(kind=k, magnitude=-0.05 if k != "volatility" else 0.5, name=k) for k in kinds]  # type: ignore[arg-type]
    # 1d path
    out = run_hypothetical_scenario(short_returns, shocks[:3])
    assert out["kind"] == "hypothetical"
    assert "returns" in out

    multi = np.column_stack([short_returns, short_returns * 1.1])
    for k in kinds:
        apply_hypothetical_shock(multi, HypotheticalShock(kind=k, magnitude=0.1, name=k))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        apply_hypothetical_shock(short_returns, HypotheticalShock(kind="nope", magnitude=1))  # type: ignore[arg-type]


def test_monte_carlo_all_methods(short_returns) -> None:
    r = short_returns
    assert bootstrap_paths(r, n_simulations=20, seed=1).shape[0] == 20
    assert block_bootstrap_paths(r, n_simulations=10, block_size=3, seed=1).shape[0] == 10
    assert trade_bootstrap_paths(r, n_simulations=10, seed=1).shape[0] == 10
    assert residual_bootstrap_paths(r, n_simulations=10, seed=1).shape[0] == 10
    assert residual_bootstrap_paths(r, fitted=r * 0.5, n_simulations=5, seed=1).shape[0] == 5
    labs = np.where(r > 0, "up", "down")
    assert regime_conditioned_paths(r, labs, n_simulations=8, seed=1).shape[0] == 8
    assert correlated_paths(r, n_simulations=8, seed=1).shape[0] == 8
    multi = np.column_stack([r, r * 0.9, r * 1.1])
    assert correlated_paths(multi, n_simulations=5, seed=2).shape[0] == 5

    for method in (
        "bootstrap",
        "block_bootstrap",
        "residual_bootstrap",
        "correlated",
    ):
        mc = run_monte_carlo(r, method=method, n_simulations=15, seed=7)
        assert mc["n_simulations"] == 15
        assert "mean_terminal" in mc

    mc_t = run_monte_carlo(r, method="trade_bootstrap", trade_pnls=r, n_simulations=10, seed=3)
    assert mc_t["method"] == "trade_bootstrap"
    mc_r = run_monte_carlo(r, method="regime_conditioned", regimes=labs, n_simulations=10, seed=3)
    assert mc_r["method"] == "regime_conditioned"

    with pytest.raises(ValueError):
        run_monte_carlo(r, method="trade_bootstrap")
    with pytest.raises(ValueError):
        run_monte_carlo(r, method="regime_conditioned")


def test_regime_liquidity_vol_corr_gap(short_returns) -> None:
    labs = classify_simple_regimes(short_returns)
    assert labs.size == short_returns.size
    reg = run_regime_scenario(short_returns, labs)
    assert reg
    rob = evaluate_regime_robustness(short_returns, labs)
    assert rob

    assert run_liquidity_scenario(short_returns, shocks=[0.25, 0.5])
    liq = apply_liquidity_shock(short_returns, shock=0.3)
    assert "returns" in liq

    assert run_volatility_scenario(short_returns, scales=[1.0, 1.5])
    vol = apply_volatility_shock(short_returns, scale=2.0)
    assert "returns" in vol

    multi = np.column_stack([short_returns, short_returns * 0.8])
    cov = np.cov(multi, rowvar=False)
    assert stress_correlation(cov, shift=0.9).shape == cov.shape
    corr = apply_correlation_shock(multi, shift=0.8, seed=1)
    assert "returns" in corr
    assert run_correlation_scenario(multi, seed=1)

    gap = apply_gap_shock(short_returns, gap=-0.1)
    assert "returns" in gap
    assert run_gap_scenario(short_returns, gaps=[-0.05, -0.1])


def test_scenario_engine(short_returns) -> None:
    eng = ScenarioEngine(n_simulations=20, seed=42)
    hist = eng.run(
        "historical",
        short_returns,
        scenario=HistoricalScenario("w", start=0, end=30),
        state={"positions": 1.0},
    )
    assert hist["state"]["positions"] == 1.0

    with pytest.raises(ValueError):
        eng.run("historical", short_returns)

    hyp = eng.run(
        "hypothetical",
        short_returns,
        shocks=[HypotheticalShock(kind="price", magnitude=-0.02)],
    )
    assert hyp["kind"] == "hypothetical"

    mc = eng.run("monte_carlo", short_returns, method="bootstrap")
    assert mc["n_simulations"] == 20

    assert eng.run("regime", short_returns)
    assert eng.run("liquidity", short_returns)
    assert eng.run("volatility", short_returns)
    multi = np.column_stack([short_returns, short_returns])
    assert eng.run("correlation", multi)
    assert eng.run("gap", short_returns)

    with pytest.raises(ValueError):
        eng.run("unknown", short_returns)

    suite = eng.run_suite(
        short_returns,
        historical=HistoricalScenario("s", start=0, end=20),
        include=["historical", "volatility", "liquidity", "gap", "monte_carlo", "hypothetical"],
    )
    assert "reports" in suite
    assert "historical" in suite["reports"]
