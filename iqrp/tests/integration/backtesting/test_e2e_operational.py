"""Complete E2E integration: synthetic → adapter → runner → … → report."""

from __future__ import annotations

from pathlib import Path

from iqrp.app.backtesting.accounting import reconcile_capital
from iqrp.app.backtesting.data import DatasetValidator, ParquetAdapter
from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv
from iqrp.app.backtesting.runner import BacktestRunConfig, BacktestRunner, RunnerLifecycleState


def test_e2e_synthetic_full_pipeline(tmp_path: Path):
    """Synthetic → adapter → runner → event engine → strategy → risk → portfolio
    → execution → fill → position → PnL → performance report → persistence.

    Does not assert profitability.
    """
    seed = 7
    data_path = tmp_path / "e2e_bars.parquet"
    write_synthetic_ohlcv(
        data_path,
        n_days=45,
        instruments=["AAA", "BBB"],
        seed=seed,
        start="2020-01-01",
    )

    adapter = ParquetAdapter(data_path, dataset_id="e2e")
    frame = adapter.load()
    report = DatasetValidator().validate(frame, raise_on_critical=True)
    assert report.ok

    cfg = BacktestRunConfig(
        backtest_id="e2e_full",
        strategy_id="buy_and_hold",
        strategy_version="1.0.0",
        strategy_params={"mode": "equal_weight"},
        dataset_path=str(data_path),
        dataset_id="e2e",
        adapter="parquet",
        start="2020-01-01",
        end="2020-02-28",
        initial_capital=1_000_000.0,
        seed=seed,
        output_dir=str(tmp_path / "results"),
        spread_bps=1.0,
        enforce_pit=True,
        risk_config={"max_gross_leverage": 1.0},
    )

    runner = BacktestRunner(cfg)
    assert runner.validate().ok
    runner.prepare()
    result = runner.run()

    assert runner.status() is RunnerLifecycleState.COMPLETED
    assert len(result.orders) > 0
    assert len(result.fills) > 0
    assert len(result.equity_curve) > 0
    assert result.capital.get("cash") is not None

    recon = reconcile_capital(result.capital, fail=False)
    assert recon.ok

    root = tmp_path / "results" / "e2e_full"
    report_md = root / "reports" / "report.md"
    report_json = root / "reports" / "report.json"
    assert report_md.exists() or report_json.exists() or Path(runner.report()).exists()
    assert (root / "reports" / "result.json").exists() or (root / "capital.json").exists()

    # Reproducibility with same seed
    out2 = tmp_path / "results2"
    cfg2 = cfg.with_updates(backtest_id="e2e_full_b", output_dir=str(out2))
    runner2 = BacktestRunner(cfg2)
    runner2.validate()
    runner2.prepare()
    result2 = runner2.run()
    assert result2.equity_curve == result.equity_curve
    assert len(result2.fills) == len(result.fills)
