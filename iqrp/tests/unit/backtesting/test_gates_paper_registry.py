"""Validation gates, paper trading, experiment registry lineage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.backtesting.engine import BacktestEngine, BacktestResult
from iqrp.app.backtesting.experiment_registry import (
    ExperimentLineage,
    ExperimentRecord,
    ExperimentRegistry,
)
from iqrp.app.backtesting.paper_trading import PaperTradingConfig, PaperTradingInterface
from iqrp.app.backtesting.performance.scorecard import StrategyScorecard
from iqrp.app.backtesting.types import BacktestState
from iqrp.app.backtesting.validation_gates import (
    GateThresholds,
    evaluate_gates,
    require_oos,
    summarize_gate_policy,
)


def test_evaluate_gates_oos_mandatory() -> None:
    # High IS Sharpe alone must NOT approve
    sc = StrategyScorecard(sharpe=3.0, total_return=1.0, out_of_sample=None, max_drawdown=0.05)
    gate = evaluate_gates(sc, in_sample_sharpe=3.0)
    assert gate.approved is False
    assert gate.out_of_sample_ok is False
    assert require_oos(sc) is False

    sc_oos = StrategyScorecard(
        sharpe=1.0,
        total_return=0.2,
        out_of_sample=0.8,
        max_drawdown=0.1,
        cvar=0.05,
        stability=0.5,
        turnover=0.2,
        transaction_costs=0.01,
        capacity=1e7,
        regime_robustness=0.5,
    )
    thr = GateThresholds(
        require_out_of_sample=True,
        min_oos_sharpe=0.0,
        min_sharpe=0.0,
        max_drawdown=0.5,
        max_cvar=0.5,
        min_stability=0.0,
        min_regime_robustness=0.0,
        max_turnover=1.0,
        max_transaction_costs=1.0,
        min_capacity=1e6,
        reject_in_sample_only=True,
    )
    gate2 = evaluate_gates(sc_oos, thr, statistical_ok=True)
    assert gate2.out_of_sample_ok is True
    assert gate2.approved is True

    # mapping form
    gate3 = evaluate_gates(sc_oos.to_dict(), thr.to_dict())
    assert gate3.out_of_sample_ok is True

    # statistical confidence required but missing
    thr2 = GateThresholds(require_out_of_sample=True, min_oos_sharpe=0.0, min_statistical_confidence=0.95)
    gate4 = evaluate_gates(sc_oos, thr2)
    assert gate4.approved is False

    assert summarize_gate_policy()
    assert GateThresholds.from_dict(None).require_out_of_sample is True
    assert gate2.to_dict()["approved"] is True


def test_paper_trading_interface(returns) -> None:
    eng = BacktestEngine()
    result = eng.run(returns=returns, seed=5, name="pt")
    iface = PaperTradingInterface()
    cfg = iface.from_result(result, gates={"approved": False})
    assert cfg.experiment_id == result.experiment_id
    assert iface.get(cfg.experiment_id) is cfg
    assert len(iface.list()) == 1
    d = cfg.to_dict()
    cfg2 = PaperTradingConfig.from_dict(d)
    assert cfg2.experiment_id == cfg.experiment_id

    # from experiment registry
    cfg3 = iface.from_experiment(result.experiment_id, eng.registry)
    assert cfg3.strategy_name == "pt"


def test_experiment_registry_lineage(tmp_path: Path) -> None:
    reg = ExperimentRegistry()
    lin = ExperimentLineage(data_version="2.0", seed=99, extra={"note": "x"})
    rec = reg.create(name="exp1", lineage=lin, config={"a": 1}, tags={"t": 1})
    assert len(reg) == 1
    assert reg.get(rec.experiment_id) is rec
    reg.update_state(rec.experiment_id, BacktestState.RUNNING)
    reg.register_result(
        rec.experiment_id,
        state=BacktestState.COMPLETED,
        metrics={"sharpe": 1.2},
        warnings=["w"],
        result_summary={"n": 10},
    )
    reg.invalidate(rec.experiment_id, "bad")
    assert reg.require(rec.experiment_id).invalidated

    # rejected retained
    listed = reg.list(include_invalidated=True)
    assert any(r.invalidated for r in listed)
    assert reg.list(include_invalidated=False) == []
    assert reg.list(state=BacktestState.INVALIDATED.value)

    path = tmp_path / "reg.json"
    reg.save(str(path))
    reg2 = ExperimentRegistry()
    n = reg2.load(str(path))
    assert n == 1

    # lineage helpers
    from iqrp.app.backtesting.config import BacktestSettings

    lin2 = ExperimentLineage.from_settings(BacktestSettings.default(), seed=7)
    assert lin2.seed == 7
    lin3 = ExperimentLineage.from_dict({"data_version": "3", "unknown": 1})
    assert lin3.extra.get("unknown") == 1
    assert ExperimentRecord.from_dict(rec.to_dict()).experiment_id == rec.experiment_id
