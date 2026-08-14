"""BacktestEngine orchestrator: run, walk_forward, scenarios, gates, save/load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.backtesting.config import BacktestSettings, CostsConfig, PITConfig
from iqrp.app.backtesting.engine import BacktestEngine, BacktestResult
from iqrp.app.backtesting.types import BacktestState
from iqrp.app.backtesting.validation_gates import GateThresholds


def test_run_basic_returns_and_signals(engine, returns, signals) -> None:
    r1 = engine.run(returns=returns, signals=signals, seed=42, name="sig")
    assert r1.state == BacktestState.COMPLETED
    assert r1.returns.size == returns.size
    assert r1.scorecard is not None
    assert r1.equity.size >= 1

    r2 = engine.run(returns=returns, signals=signals, seed=42, name="sig")
    np.testing.assert_allclose(r1.returns, r2.returns)


def test_run_prices_strategy_fn_signal_fn(engine, prices, returns) -> None:
    def strategy_fn(t, history):
        return {"weight": 1.0 if history[-1] > 0 else -0.5}

    r = engine.run(prices=prices, strategy_fn=strategy_fn, seed=1, costs=True)
    assert r.state == BacktestState.COMPLETED
    assert r.trades  # turnover from sign flips

    def signal_fn(t, history=None):
        return 0.5

    r2 = engine.run(returns=returns[:40], signal_fn=signal_fn, seed=2, execution_sim=False)
    assert r2.state == BacktestState.COMPLETED

    # default flat long
    r3 = engine.run(returns=returns[:30], seed=3, costs=False)
    assert r3.state == BacktestState.COMPLETED


def test_run_multi_asset_prices_and_oos(engine, multi_prices) -> None:
    r = engine.run(prices=multi_prices, oos_fraction=0.25, seed=5)
    assert r.oos_returns is not None
    assert r.oos_returns.size > 0


def test_run_invalidate_leakage_and_universe(settings) -> None:
    settings = BacktestSettings(
        name="pit",
        pit=PITConfig(detect_leakage=True),
        costs=CostsConfig(),
    )
    eng = BacktestEngine(settings=settings)
    rets = np.random.default_rng(0).normal(0, 0.01, size=20)
    bad = eng.run(
        returns=rets,
        feature_asof_index=[0, 1, 2],
        label_asof_index=[0, 1, 5],
        seed=1,
    )
    # timestamps length = 20, label index 5 is ok for length; need label > feature
    assert bad.state == BacktestState.INVALIDATED or bad.invalidated

    membership = {"A": (0, 10), "B": (5, None)}
    # universe_asof with membership — ok path
    ok = eng.run(returns=rets, membership=membership, universe_asof=5, seed=2)
    assert ok.state in (BacktestState.COMPLETED, BacktestState.INVALIDATED)


def test_run_failed_without_data(engine) -> None:
    r = engine.run(seed=1)
    assert r.state == BacktestState.FAILED


def test_walk_forward_retrain_scenarios(engine, returns) -> None:
    wf = engine.walk_forward(
        returns=returns, train_size=60, test_size=20, step=20, validation_size=0
    )
    assert wf["n_folds"] >= 1

    X = np.column_stack([returns, np.roll(returns, 1)])
    y = returns
    rr = engine.retrain_rolling(X=X, y=y, every=40)
    assert rr["n_models"] >= 1

    engine.run(returns=returns, seed=9)
    sc = engine.scenarios("monte_carlo", n_simulations=15, seed=9)
    assert sc["n_simulations"] == 15

    with pytest.raises(ValueError):
        BacktestEngine().scenarios("monte_carlo")


def test_capacity_sweep_ablation_compare_scorecard(engine, returns) -> None:
    engine.run(returns=returns, seed=1)
    cap = engine.capacity_test(capital_levels=np.array([1e6, 1e7, 1e8]))
    assert "curve" in cap and "limit" in cap

    def obj(lookback=10, **flags):
        lb = int(lookback)
        r = returns[lb:] - returns[:-lb].mean() * 0.01
        if flags.get("use_signal", True) is False:
            r = r * 0.5
        return r

    sweep = engine.parameter_sweep(obj, {"lookback": [5, 10, 15]})
    assert sweep["n_combinations"] == 3

    abl = engine.ablation(obj, components={"use_signal": True})
    assert abl["name"] == "ablation"

    sens = engine.sensitivity(obj, {"lookback": 10})
    assert sens["name"] == "sensitivity"

    cmp = engine.compare({"a": returns, "b": returns * 0.5})
    assert cmp

    result = engine.run(returns=returns, seed=2)
    sc = engine.scorecard(result)
    assert sc.sharpe == result.scorecard.sharpe
    sc2 = engine.scorecard({"returns": returns, "scorecard": result.scorecard.to_dict()})
    assert sc2.sharpe == result.scorecard.sharpe


def test_validate_for_promotion_requires_oos(engine, returns) -> None:
    # High IS sharpe without OOS must fail
    result = engine.run(returns=returns, seed=1)  # no oos_fraction
    gate = engine.validate_for_promotion(result)
    assert gate.approved is False
    assert gate.out_of_sample_ok is False

    with_oos = engine.run(returns=returns, oos_fraction=0.3, seed=2)
    gate2 = engine.validate_for_promotion(
        with_oos,
        gates=GateThresholds(
            require_out_of_sample=True,
            min_oos_sharpe=-100.0,
            max_drawdown=1.0,
            min_sharpe=-100.0,
        ),
    )
    # May or may not approve depending on metrics, but OOS ok should be True
    assert gate2.out_of_sample_ok is True

    # invalidated cannot promote
    with_oos.invalidated = True
    with_oos.invalidation_reason = "leak"
    gate3 = engine.validate_for_promotion(with_oos)
    assert gate3.approved is False


def test_paper_trading_save_load_invalidate(engine, returns, tmp_path: Path) -> None:
    result = engine.run(returns=returns, oos_fraction=0.25, seed=11, name="paper_me")
    pt = engine.to_paper_trading(result)
    assert pt.experiment_id == result.experiment_id
    assert pt.seed == 11

    pt2 = engine.to_paper_trading(result.experiment_id)
    assert pt2.experiment_id == result.experiment_id

    path = tmp_path / "bt.json"
    engine.save(path, result)
    eng2 = BacktestEngine()
    loaded = eng2.load(path)
    assert loaded.experiment_id == result.experiment_id
    np.testing.assert_allclose(loaded.returns, result.returns)

    engine.invalidate(result.experiment_id, "manual")
    assert engine._last_result.invalidated

    with pytest.raises(ValueError):
        BacktestEngine().save(tmp_path / "x.json")
    with pytest.raises(ValueError):
        BacktestEngine().to_paper_trading()


def test_backtest_result_roundtrip() -> None:
    r = BacktestResult(
        experiment_id="abc", state=BacktestState.COMPLETED, returns=np.array([0.01, -0.02])
    )
    d = r.to_dict()
    r2 = BacktestResult.from_dict(d)
    assert r2.experiment_id == "abc"
    r3 = BacktestResult.from_dict({"experiment_id": "x", "state": "NOT_A_STATE"})
    assert r3.state == BacktestState.FAILED


def test_settings_from_mapping_and_hydra() -> None:
    s = BacktestSettings.from_mapping({"name": "h", "initial_cash": 5000.0})
    assert s.name == "h"
    s2 = BacktestSettings.default()
    assert s2.enabled
    s3 = BacktestSettings.from_hydra()
    assert s3 is not None
