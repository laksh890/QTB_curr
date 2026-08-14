"""Reports, serializer, and default component registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.backtesting.engine import BacktestEngine
from iqrp.app.backtesting.registry import BacktestingRegistry, ComponentSpec, default_registry
from iqrp.app.backtesting.reports import (
    attribution_report,
    capacity_report,
    cost_report,
    drawdown_report,
    execution_report,
    exposure_report,
    full_report,
    performance_report,
    risk_report,
    scenario_report,
    scorecard_report,
    sensitivity_report,
    trade_report,
)
from iqrp.app.backtesting.serializer import (
    deserialize_result,
    load_json,
    save_json,
    serialize_result,
    to_jsonable,
)
from iqrp.app.backtesting.types import BacktestState


def test_full_report_and_sections(returns, trade_list) -> None:
    eng = BacktestEngine()
    result = eng.run(returns=returns, seed=3, name="rep")
    report = full_report(result)
    assert report["experiment_id"] == result.experiment_id
    assert "performance" in report
    assert "scorecard" in report

    assert performance_report(returns)["name"] == "performance"
    assert risk_report(returns)["name"] == "risk"
    assert drawdown_report(returns)["name"] == "drawdown"
    assert trade_report(trade_list, positions=result.exposures)["name"] == "trades"
    assert exposure_report(result.exposures)["name"] == "exposure"
    assert attribution_report(returns)["note"]
    factors = np.column_stack([returns, np.roll(returns, 1)])
    assert attribution_report(returns, factors=factors, factor_exposures=np.ones_like(factors))
    assert cost_report(result.costs)["name"] == "costs"
    assert execution_report([{"qty": 1}], latency={"ms": 1})["n_fills"] == 1
    assert scenario_report({"mc": {}})["name"] == "scenarios"
    assert capacity_report({"limit": 1})["name"] == "capacity"
    assert sensitivity_report({"x": 1})["name"] == "sensitivity"
    assert scorecard_report(returns, oos_returns=returns[-50:])["name"] == "scorecard"


def test_serialize_deserialize(returns, tmp_path: Path) -> None:
    eng = BacktestEngine()
    result = eng.run(returns=returns, seed=4)
    payload = serialize_result(result)
    assert payload["experiment_id"] == result.experiment_id
    restored = deserialize_result(payload)
    assert restored.experiment_id == result.experiment_id
    np.testing.assert_allclose(restored.returns, result.returns)

    path = tmp_path / "out.json"
    save_json(path, {"a": np.array([1, 2]), "s": BacktestState.COMPLETED})
    data = load_json(path)
    assert data["a"] == [1, 2]
    assert data["s"] == "COMPLETED"

    assert to_jsonable(None) is None
    assert to_jsonable(np.float64(1.5)) == 1.5
    assert to_jsonable({"k": (1, 2)}) == {"k": [1, 2]}

    # mapping serialize
    assert serialize_result({"experiment_id": "x", "returns": [0.1]})["experiment_id"] == "x"


def test_default_registry() -> None:
    reg = default_registry()
    assert len(reg.list()) >= 10
    assert reg.get("Backtest Engine") is not None
    assert reg.list(category="engine")
    assert reg.to_list()[0]["name"]

    spec = ComponentSpec(
        "T", "test", "iqrp.app.backtesting", "BacktestEngine", ["BacktestingPlatform.md"]
    )
    custom = BacktestingRegistry([spec])
    custom.register(ComponentSpec("U", "test", "iqrp.app.backtesting", "BacktestSettings"))
    assert custom.get("T").to_dict()["symbol"] == "BacktestEngine"
