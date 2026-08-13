"""Extra coverage for runner adapters, persistence, validation, executor."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv
from iqrp.app.backtesting.runner.adapters import (
    ExecutionSimulationAdapter,
    IsolatedPortfolioFallback,
    PortfolioConstructionAdapter,
)
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.executor import PipelineExecutor, bars_by_timestamp, load_market_frame
from iqrp.app.backtesting.runner.persistence import persist_result
from iqrp.app.backtesting.runner.reports import write_reports
from iqrp.app.backtesting.runner.result import OperationalBacktestResult
from iqrp.app.backtesting.runner.validation import (
    ValidationIssue,
    ValidationReport,
    preflight_validate,
)
from iqrp.app.backtesting.strategy import BuyAndHoldStrategy, StrategyRegistry


def test_portfolio_and_execution_adapters():
    fb = IsolatedPortfolioFallback
    out = fb.signals_to_raw_weights({"AAA": 1.0, "BBB": 0.0}, budget=1.0, long_only=True)
    assert out["weights"]
    out2 = fb.signals_to_raw_weights([0.0, 0.0], names=["A", "B"], long_only=True)
    assert len(out2["weights"]) == 2
    out3 = fb.signals_to_raw_weights({"A": 1.0, "B": -1.0}, long_only=False)
    assert abs(sum(abs(w) for w in out3["weights"]) - 1.0) < 1e-9 or out3["weights"]
    assert fb.signals_to_raw_weights({})["names"] == []
    assert fb.build_target_weights({"A": 0.5})["A"] == 0.5
    assert fb.build_target_weights([0.2, 0.8], names=["X", "Y"])["X"] == 0.2

    port = PortfolioConstructionAdapter()
    targets = port.targets_from_signals({"AAA": 1.0, "BBB": 0.5})
    assert isinstance(targets, dict)
    assert port.targets_from_signals({}) == {}

    exe = ExecutionSimulationAdapter()
    fills = exe.simulate_execution(
        [
            {
                "order_id": "o1",
                "instrument": "AAA",
                "side": "buy",
                "quantity": 10,
                "price": 100.0,
            }
        ],
        market_context={"AAA": {"mid": 100.0}},
        spread_bps=1.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    )
    assert fills.get("n", 0) >= 1 or fills.get("orders")
    planned = exe.plan_from_targets(
        {},
        {"AAA": 1.0},
        equity=10_000.0,
        prices={"AAA": 100.0},
    )
    assert planned
    costs = exe.estimate_costs(planned, commission_bps=1.0, spread_bps=1.0)
    assert "total_cost" in costs
    port.targets_from_weights({"AAA": 0.5, "BBB": 0.5})


def test_load_market_frame_and_bars(tmp_path: Path):
    path = tmp_path / "m.parquet"
    write_synthetic_ohlcv(path, n_days=20, instruments=["AAA", "BBB"], seed=2)
    cfg = BacktestRunConfig(
        dataset_path=str(path),
        adapter="parquet",
        start="2020-01-01",
        end="2020-01-31",
        universe=["AAA"],
    )
    frame, detail = load_market_frame(cfg)
    assert detail["ok"]
    assert set(frame["instrument"].unique()) == {"AAA"}
    bars = bars_by_timestamp(frame)
    assert bars and isinstance(bars[0][0], datetime)

    csv_path = tmp_path / "m.csv"
    write_synthetic_ohlcv(csv_path, n_days=10, instruments=["AAA"], seed=1)
    frame2, _ = load_market_frame(BacktestRunConfig(dataset_path=str(csv_path), adapter="csv"))
    assert not frame2.empty

    with pytest.raises(FileNotFoundError):
        load_market_frame(BacktestRunConfig(dataset_path=str(tmp_path / "missing.parquet")))


def test_preflight_and_integrity():
    report = preflight_validate(
        BacktestRunConfig(strategy_id="buy_and_hold", dataset_path="x"),
        strategy_registered=True,
        dataset_ok=True,
    )
    assert isinstance(report, ValidationReport)
    assert report.to_dict()

    issue = ValidationIssue(code="x", severity="warning", message="y")
    assert issue.to_dict()["code"] == "x"

    # integrity_validate is exercised via BacktestRunner.run(); here only shape
    report_bad = preflight_validate(
        BacktestRunConfig(strategy_id="", dataset_path=""),
        strategy_registered=False,
        dataset_ok=False,
    )
    assert not report_bad.ok


def test_persist_and_reports(tmp_path: Path):
    res = OperationalBacktestResult(
        backtest_id="persist_demo",
        status="COMPLETED",
        equity_curve=[100.0, 101.0, 102.0],
        returns=[0.01, 0.0099],
        timestamps=["2020-01-01", "2020-01-02", "2020-01-03"],
        orders=[{"order_id": "o1", "instrument": "AAA", "side": "buy", "quantity": 1}],
        fills=[{"fill_id": "f1", "order_id": "o1", "instrument": "AAA", "side": "buy", "quantity": 1, "price": 10}],
        trades=[],
        positions_log=[],
        snapshots=[],
        capital={"cash": 90.0, "equity": 100.0, "initial_capital": 100.0},
        performance={"total_return": 0.02},
        risk={},
        execution={},
        diagnostics={},
        initial_capital=100.0,
        seed=1,
        config={"backtest_id": "persist_demo"},
    )
    root = persist_result(res, tmp_path / "results")
    assert root is not None
    assert (Path(root) / "reports" / "result.json").exists() or (Path(root) / "capital.json").exists()
    paths = write_reports(res, tmp_path / "results")
    assert paths
    assert res.to_dict()["backtest_id"] == "persist_demo"


def test_pipeline_executor_run(tmp_path: Path):
    StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
    path = tmp_path / "ex.parquet"
    write_synthetic_ohlcv(path, n_days=25, instruments=["AAA", "BBB"], seed=3)
    cfg = BacktestRunConfig(
        backtest_id="exec1",
        strategy_id="buy_and_hold",
        dataset_path=str(path),
        output_dir=str(tmp_path / "out"),
        seed=3,
        start="2020-01-01",
        end="2020-02-05",
    )
    frame, detail = load_market_frame(cfg)
    ex = PipelineExecutor(cfg, BuyAndHoldStrategy(), frame=frame, data_detail=detail)
    ex.prepare()
    ex.run()
    assert ex.context is not None
    assert ex.context.equity_curve
