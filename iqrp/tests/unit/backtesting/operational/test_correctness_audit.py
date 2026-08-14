"""Correctness audit regression: status consistency, recon, NIFTY, reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv
from iqrp.app.backtesting.runner import BacktestRunner
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.lifecycle import RunnerLifecycleState
from iqrp.app.backtesting.strategy import (
    BuyAndHoldStrategy,
    StrategyRegistry,
)

NIFTY_PATH = Path("data/nifty50/nifty50_daily.parquet")


@pytest.fixture(autouse=True)
def _register_strategies() -> None:
    try:
        StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
    except Exception:
        pass


def _run_synth(tmp_path: Path, *, backtest_id: str, seed: int = 7) -> tuple[BacktestRunner, Path]:
    data = tmp_path / "bars.parquet"
    write_synthetic_ohlcv(data, n_days=40, instruments=["AAA", "BBB"], seed=seed)
    out = tmp_path / "results"
    cfg = BacktestRunConfig(
        backtest_id=backtest_id,
        strategy_id="buy_and_hold",
        strategy_version="1.0.0",
        dataset_path=str(data),
        adapter="parquet",
        start="2020-01-01",
        end="2020-02-15",
        initial_capital=1_000_000.0,
        seed=seed,
        output_dir=str(out),
        strategy_params={"mode": "equal_weight"},
    )
    runner = BacktestRunner(cfg)
    runner.validate()
    runner.prepare()
    runner.run()
    return runner, out / backtest_id


def test_status_consistency_completed_persisted(tmp_path: Path) -> None:
    runner, root = _run_synth(tmp_path, backtest_id="status_audit")
    assert runner.status() is RunnerLifecycleState.COMPLETED
    assert runner.result().status == "COMPLETED"

    result_json = json.loads((root / "reports" / "result.json").read_text(encoding="utf-8"))
    report_json = json.loads((root / "reports" / "report.json").read_text(encoding="utf-8"))
    report_md = (root / "reports" / "report.md").read_text(encoding="utf-8")
    diagnostics = json.loads(
        (root / "diagnostics" / "diagnostics.json").read_text(encoding="utf-8")
    )
    recon_file = json.loads(
        (root / "diagnostics" / "reconciliation.json").read_text(encoding="utf-8")
    )

    assert result_json["status"] == "COMPLETED"
    assert report_json["executive_summary"]["status"] == "COMPLETED"
    assert "Status: `COMPLETED`" in report_md
    assert diagnostics.get("status") == "COMPLETED"
    assert result_json.get("reconciliation", {}).get("ok") is True
    assert diagnostics.get("reconciliation", {}).get("ok") is True
    assert recon_file.get("ok") is True


def test_reconciliation_always_present_and_ok(tmp_path: Path) -> None:
    runner, root = _run_synth(tmp_path, backtest_id="recon_audit")
    res = runner.result()
    assert isinstance(res.reconciliation, dict)
    assert res.reconciliation.get("ok") is True
    assert "cash" in res.reconciliation or "ending_equity" in res.reconciliation
    assert res.diagnostics.get("reconciliation", {}).get("ok") is True

    disk = json.loads((root / "reports" / "result.json").read_text(encoding="utf-8"))
    assert disk["reconciliation"]["ok"] is True
    assert disk["diagnostics"]["reconciliation"]["ok"] is True


def test_result_json_completeness(tmp_path: Path) -> None:
    _, root = _run_synth(tmp_path, backtest_id="complete_audit")
    data = json.loads((root / "reports" / "result.json").read_text(encoding="utf-8"))
    required = [
        "backtest_id",
        "status",
        "initial_capital",
        "ending_equity",
        "equity_curve",
        "orders",
        "fills",
        "trades",
        "positions",
        "performance",
        "risk",
        "diagnostics",
        "reconciliation",
        "configuration",
        "dataset",
        "strategy",
    ]
    for key in required:
        assert key in data, f"missing {key}"
    assert data["status"] == "COMPLETED"
    assert len(data["equity_curve"]) > 0


def test_artifact_consistency(tmp_path: Path) -> None:
    runner, root = _run_synth(tmp_path, backtest_id="artifact_audit")
    result = json.loads((root / "reports" / "result.json").read_text(encoding="utf-8"))
    report = json.loads((root / "reports" / "report.json").read_text(encoding="utf-8"))
    diag = json.loads((root / "diagnostics" / "diagnostics.json").read_text(encoding="utf-8"))
    recon = json.loads((root / "diagnostics" / "reconciliation.json").read_text(encoding="utf-8"))
    md = (root / "reports" / "report.md").read_text(encoding="utf-8")

    assert result["status"] == report["executive_summary"]["status"] == "COMPLETED"
    assert result["status"] == diag["status"]
    assert (
        abs(float(result["ending_equity"]) - float(report["executive_summary"]["ending_equity"]))
        < 1e-8
    )
    assert len(result["orders"]) == report["executive_summary"]["n_orders"]
    assert len(result["fills"]) == report["executive_summary"]["n_fills"]
    assert result["reconciliation"]["ok"] is True
    assert recon["ok"] is True
    assert runner.status().value == "COMPLETED"
    assert "COMPLETED" in md


def test_synthetic_reproducibility(tmp_path: Path) -> None:
    r1, root1 = _run_synth(tmp_path / "a", backtest_id="repro_a", seed=42)
    r2, root2 = _run_synth(tmp_path / "b", backtest_id="repro_b", seed=42)
    a = r1.result()
    b = r2.result()
    assert a.equity_curve == b.equity_curve
    assert a.orders == b.orders
    assert a.fills == b.fills
    assert abs(a.equity_curve[-1] - b.equity_curve[-1]) < 1e-12
    assert a.performance.get("total_return") == b.performance.get("total_return")
    da = json.loads((root1 / "reports" / "result.json").read_text(encoding="utf-8"))
    db = json.loads((root2 / "reports" / "result.json").read_text(encoding="utf-8"))
    assert da["equity_curve"] == db["equity_curve"]
    assert da["ending_equity"] == db["ending_equity"]


@pytest.mark.skipif(not NIFTY_PATH.exists(), reason="NIFTY parquet not present locally")
def test_nifty_real_data_regression(tmp_path: Path) -> None:
    out = tmp_path / "nifty_results"
    cfg = BacktestRunConfig(
        backtest_id="nifty_audit_regression",
        strategy_id="buy_and_hold",
        strategy_version="1.0.0",
        dataset_path=str(NIFTY_PATH),
        adapter="parquet",
        start="2018-01-01",
        end="2023-12-31",
        initial_capital=1_000_000.0,
        seed=42,
        output_dir=str(out),
        currency="INR",
        strategy_params={"mode": "equal_weight"},
    )
    runner = BacktestRunner(cfg)
    runner.validate()
    runner.prepare()
    runner.run()

    assert runner.status() is RunnerLifecycleState.COMPLETED
    res = runner.result()
    assert res.status == "COMPLETED"
    assert res.diagnostics.get("data_validated") is True
    assert res.reconciliation.get("ok") is True
    assert res.diagnostics.get("reconciliation", {}).get("ok") is True
    assert len(res.orders) == 1
    assert len(res.fills) == 1
    assert len(res.equity_curve) > 0
    assert float(res.equity_curve[-1]) == float(res.equity_curve[-1])  # finite
    assert all(x == x for x in res.equity_curve)

    root = out / "nifty_audit_regression"
    disk = json.loads((root / "reports" / "result.json").read_text(encoding="utf-8"))
    assert disk["status"] == "COMPLETED"
    assert disk["reconciliation"]["ok"] is True
    assert disk["diagnostics"]["reconciliation"]["ok"] is True
    assert len(disk["orders"]) == 1
    assert len(disk["fills"]) == 1

    # Reproducibility on real data
    out2 = tmp_path / "nifty_results_b"
    cfg2 = cfg.with_updates(backtest_id="nifty_audit_regression_b", output_dir=str(out2))
    r2 = BacktestRunner(cfg2)
    r2.validate()
    r2.prepare()
    r2.run()
    assert r2.result().equity_curve == res.equity_curve
    assert r2.result().orders == res.orders
    assert r2.result().fills == res.fills
