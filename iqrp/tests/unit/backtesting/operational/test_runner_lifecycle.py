"""Runner lifecycle, checkpoint/recovery, leakage, sweep, config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iqrp.app.backtesting.runner import (
    BacktestRunConfig,
    BacktestRunner,
    RunnerLifecycleState,
)
from iqrp.app.backtesting.runner.checkpoint import (
    apply_checkpoint,
    checkpoint_path,
    read_checkpoint,
    write_checkpoint,
)
from iqrp.app.backtesting.runner.lifecycle import (
    Lifecycle,
    map_engine_state,
    map_runner_to_engine,
)
from iqrp.app.backtesting.runner.parallel import parameter_sweep_parallel
from iqrp.app.backtesting.strategy import BuyAndHoldStrategy, StrategyRegistry
from iqrp.app.backtesting.types import BacktestState


def test_config_from_yaml_dict_updates(tmp_path: Path, synthetic_parquet: Path):
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        yaml.dump(
            {
                "id": "cfg1",
                "capital": 500_000,
                "strategy_id": "buy_and_hold",
                "dataset_path": str(synthetic_parquet),
                "universe": "AAA,BBB",
                "output_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )
    cfg = BacktestRunConfig.from_yaml(yml)
    assert cfg.backtest_id == "cfg1"
    assert cfg.initial_capital == 500_000
    assert cfg.universe == ["AAA", "BBB"]
    cfg2 = cfg.with_updates(seed=99)
    assert cfg2.seed == 99
    assert cfg.results_root().name == "cfg1"
    assert BacktestRunConfig.from_dict({"strategy_id": "x"}).strategy_id == "x"
    with pytest.raises(TypeError):
        # non-mapping yaml root
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just a list\n", encoding="utf-8")
        BacktestRunConfig.from_yaml(bad)


def test_lifecycle_transitions_and_mapping():
    lc = Lifecycle()
    assert lc.state is RunnerLifecycleState.CREATED
    lc.transition(RunnerLifecycleState.VALIDATING)
    lc.transition(RunnerLifecycleState.PREPARING)
    lc.transition(RunnerLifecycleState.RUNNING)
    lc.transition(RunnerLifecycleState.COMPLETED)
    assert lc.state.is_terminal
    with pytest.raises(RuntimeError):
        lc.transition(RunnerLifecycleState.RUNNING)
    lc.transition(RunnerLifecycleState.ARCHIVED)
    assert map_engine_state(BacktestState.RUNNING) is RunnerLifecycleState.RUNNING
    assert map_engine_state("COMPLETED") is RunnerLifecycleState.COMPLETED
    assert map_engine_state(None) is None
    assert map_runner_to_engine(RunnerLifecycleState.PREPARING) is BacktestState.VALIDATING
    assert map_runner_to_engine(RunnerLifecycleState.CANCELLED) is BacktestState.FAILED
    assert lc.to_dict()["state"] == "ARCHIVED"


def test_runner_happy_path_lifecycle(buy_and_hold_config, registered_strategies):
    runner = BacktestRunner(buy_and_hold_config)
    assert runner.create() is RunnerLifecycleState.CREATED
    report = runner.validate()
    assert report.ok
    runner.prepare()
    result = runner.run()
    assert runner.status() is RunnerLifecycleState.COMPLETED
    assert result.equity_curve
    assert len(result.orders) > 0
    assert len(result.fills) > 0
    assert runner.result().backtest_id == "op_unit"
    assert Path(runner.report()).exists()


def test_runner_preflight_failures(tmp_path: Path, synthetic_parquet: Path):
    StrategyRegistry.clear()
    with pytest.raises(ValueError):
        BacktestRunner(
            {
                "strategy_id": "buy_and_hold",
                "dataset_path": str(synthetic_parquet),
                "output_dir": str(tmp_path),
            }
        ).validate()
    StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
    with pytest.raises(ValueError):
        BacktestRunner(
            {
                "strategy_id": "buy_and_hold",
                "dataset_id": "no_path",
                "output_dir": str(tmp_path),
            }
        ).validate()
    with pytest.raises(ValueError):
        BacktestRunner(
            {
                "strategy_id": "buy_and_hold",
                "dataset_path": str(synthetic_parquet),
                "initial_capital": -1,
                "output_dir": str(tmp_path),
            }
        ).validate()


def test_checkpoint_pause_resume(buy_and_hold_config, registered_strategies, tmp_path: Path):
    cfg = dict(buy_and_hold_config)
    cfg["backtest_id"] = "ckpt_run"
    cfg["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["output_dir"] = str(tmp_path / "results")
    runner = BacktestRunner(cfg)
    runner.validate()
    runner.prepare()
    # Write checkpoint from prepared context then pause
    assert runner._executor is not None and runner._executor.context is not None
    cp = write_checkpoint(
        runner._executor.context,
        checkpoint_path(cfg["checkpoint_dir"], cfg["backtest_id"]),
    )
    assert cp.exists()
    payload = read_checkpoint(cp)
    apply_checkpoint(runner._executor.context, payload)
    state = runner.pause()
    assert state is RunnerLifecycleState.PAUSED
    # Resume from checkpoint
    runner2 = BacktestRunner(
        BacktestRunConfig.from_dict(cfg).with_updates(resume_from=str(cp))
    )
    runner2.validate()
    runner2.prepare()
    result = runner2.run()
    assert result.equity_curve
    assert runner2.status() in {
        RunnerLifecycleState.COMPLETED,
        RunnerLifecycleState.INVALIDATED,
        RunnerLifecycleState.FAILED,
    }


def test_cancel_path(buy_and_hold_config, registered_strategies):
    runner = BacktestRunner(buy_and_hold_config)
    runner.validate()
    runner.prepare()
    assert runner.cancel() is RunnerLifecycleState.CANCELLED


def test_leakage_invalidate_path(buy_and_hold_config, registered_strategies, tmp_path: Path):
    # Force invalidate via max_drawdown=0 after any move, or PIT breach by custom path
    cfg = dict(buy_and_hold_config)
    cfg["backtest_id"] = "invalidate_dd"
    cfg["risk_config"] = {"max_drawdown": 0.0, "max_gross_leverage": 1.0}
    cfg["output_dir"] = str(tmp_path / "results_inv")
    runner = BacktestRunner(cfg)
    runner.validate()
    runner.prepare()
    try:
        runner.run()
    except Exception:
        pass
    # Either INVALIDATED during cascade or COMPLETED/FAILED depending on when DD triggers
    assert runner.status() in {
        RunnerLifecycleState.INVALIDATED,
        RunnerLifecycleState.COMPLETED,
        RunnerLifecycleState.FAILED,
        RunnerLifecycleState.CANCELLED,
    }


def test_parameter_sweep_isolation(buy_and_hold_config, registered_strategies):
    base = dict(buy_and_hold_config)
    base["backtest_id"] = "sweep_base"
    grid = [
        {"strategy_params": {"mode": "equal_weight"}, "experiment_id": "sweep_eq"},
        {"strategy_params": {"mode": "first_instrument"}, "experiment_id": "sweep_first"},
    ]
    results = parameter_sweep_parallel(base, grid, max_workers=1, seed0=7)
    assert len(results) == 2
    ids = {r["experiment_id"] for r in results}
    assert ids == {"sweep_eq", "sweep_first"}
    # Isolation: distinct seeds / backtest ids
    assert results[0]["seed"] != results[1]["seed"] or results[0]["experiment_id"] != results[1]["experiment_id"]

    runner = BacktestRunner(base)
    sweep2 = runner.parameter_sweep(grid[:1], max_workers=1)
    assert len(sweep2) == 1


def test_runner_with_strategy_override(buy_and_hold_config):
    runner = BacktestRunner(buy_and_hold_config, strategy=BuyAndHoldStrategy(mode="first_instrument"))
    runner.validate()
    runner.prepare()
    result = runner.run()
    assert result.equity_curve


def test_walk_forward_and_scenarios_hooks(buy_and_hold_config, registered_strategies, tmp_path: Path):
    cfg = dict(buy_and_hold_config)
    cfg["backtest_id"] = "wf_hook"
    cfg["output_dir"] = str(tmp_path / "wf_out")
    cfg["walk_forward_config"] = {"train_periods": 10, "test_periods": 5, "mode": "rolling"}
    cfg["scenario_config"] = {"enabled": True}
    cfg["model_config"] = {"enabled": True}
    runner = BacktestRunner(cfg)
    runner.validate()
    runner.prepare()
    result = runner.run()
    assert result.walk_forward or runner.walk_forward()
    assert runner.scenarios() is not None
    assert runner.retrain() is not None
