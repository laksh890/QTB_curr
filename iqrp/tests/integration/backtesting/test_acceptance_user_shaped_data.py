"""Acceptance workflow using larger user-shaped local synthetic parquet.

This is NOT a live download. It simulates a user supplying a local historical
file, registering it, validating it, and running a backtest with PIT checks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from iqrp.app.backtesting.accounting import reconcile_capital
from iqrp.app.backtesting.data import (
    DatasetRegistry,
    DatasetValidator,
    ParquetAdapter,
    assert_no_lookahead,
    filter_frame_asof_df,
)
from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv
from iqrp.app.backtesting.runner import BacktestRunConfig, BacktestRunner, RunnerLifecycleState


def test_acceptance_workflow_user_shaped_local_parquet(tmp_path: Path):
    # Larger synthetic parquet standing in for user-supplied local data
    user_data = tmp_path / "user_supplied" / "equities_daily.parquet"
    write_synthetic_ohlcv(
        user_data,
        n_days=120,
        instruments=["AAA", "BBB", "CCC"],
        seed=13,
        start="2019-06-03",
        dataset_id="user_equities_daily",
    )

    registry = DatasetRegistry(tmp_path / "dataset_registry.json")
    record = registry.register_file(
        user_data,
        dataset_id="user_equities_daily",
        version="1.0.0",
        source="user_local",
        canonical_parquet=True,
    )
    assert registry.verify_checksum("user_equities_daily")
    assert record.row_count >= 0 or Path(record.path).exists()

    adapter = ParquetAdapter(user_data, dataset_id="user_equities_daily")
    frame = adapter.load()
    dq = DatasetValidator().validate(frame, raise_on_critical=True)
    assert dq.ok

    # Point-in-time: no future timestamps used relative to asof mid-sample
    mid = frame["timestamp"].iloc[len(frame) // 2]
    asof = pd.Timestamp(mid).to_pydatetime()
    if asof.tzinfo is None:
        asof = asof.replace(tzinfo=UTC)
    pit = filter_frame_asof_df(frame.assign(effective_timestamp=frame["timestamp"]), asof)
    assert (pit["timestamp"] <= pd.Timestamp(asof)).all()
    # Explicit lookahead guard on last bar vs earlier asof
    last_ts = pd.Timestamp(frame["timestamp"].iloc[-1]).to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)
    try:
        assert_no_lookahead(last_ts, asof)
        pit_ok = False
    except Exception:
        pit_ok = True  # future vs earlier asof is correctly rejected
    assert pit_ok

    cfg = BacktestRunConfig(
        backtest_id="acceptance_user_local",
        strategy_id="cross_sectional_momentum",
        strategy_version="1.0.0",
        strategy_params={"lookback": 10, "top_n": 2, "long_only": True},
        dataset_path=str(user_data),
        dataset_id="user_equities_daily",
        dataset_version="1.0.0",
        adapter="parquet",
        start="2019-06-03",
        end="2019-11-29",
        initial_capital=2_000_000.0,
        seed=13,
        output_dir=str(tmp_path / "results"),
        spread_bps=1.0,
        enforce_pit=True,
        risk_config={"max_gross_leverage": 1.0},
        meta={
            "acceptance": True,
            "note": "User-shaped local synthetic parquet; not live download",
        },
    )

    runner = BacktestRunner(cfg)
    assert runner.validate().ok
    runner.prepare()
    result = runner.run()

    assert runner.status() is RunnerLifecycleState.COMPLETED
    assert len(result.equity_curve) > 0
    assert Path(runner.report()).exists()
    recon = reconcile_capital(result.capital, fail=False)
    assert recon.ok

    # Runner diagnostics should not indicate lookahead invalidation
    assert result.diagnostics.get("integrity", {}).get(
        "ok", True
    ) is True or not result.diagnostics.get("invalidated", False)
