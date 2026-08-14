"""Integration: data → feature → signal → position → research report (+ optional runner)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iqrp.app.backtesting.alpha_research.engine import AlphaSignalResearchEngine
from iqrp.app.backtesting.alpha_research.experiments import ExperimentRegistry
from iqrp.app.backtesting.data.dataset_registry import compute_checksum
from iqrp.app.backtesting.strategy.long_short_momentum import LongShortMomentumStrategy
from iqrp.app.backtesting.strategy.registry import StrategyRegistry
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.runner import BacktestRunner


NIFTY = {
    "5m": Path("data/nifty50/nifty50_intraday_5m.parquet"),
    "15m": Path("data/nifty50/nifty50_intraday_15m.parquet"),
    "30m": Path("data/nifty50/nifty50_intraday_30m.parquet"),
    "1h": Path("data/nifty50/nifty50_intraday_1h.parquet"),
}


@pytest.fixture
def nifty_frames():
    missing = [k for k, p in NIFTY.items() if not p.exists()]
    if missing:
        pytest.skip(f"missing nifty intraday: {missing}")
    return {k: pd.read_parquet(p) for k, p in NIFTY.items()}


def test_research_matrix_nifty_dev(nifty_frames, tmp_path):
    checksums = {k: compute_checksum(NIFTY[k]) for k in nifty_frames}
    eng = AlphaSignalResearchEngine(experiment_registry=ExperimentRegistry(tmp_path / "exp.json"))
    report = eng.run_matrix(
        nifty_frames,
        signal_ids=["momentum_signal", "mean_reversion_signal", "breakout_signal"],
        timeframes=["5m", "15m", "30m", "1h"],
        holding_bars=[3, 5],
        lookbacks=[10, 20],
        dataset_checksums=checksums,
    )
    assert report["n_experiments"] > 0
    assert all(row.get("sample_flag") == "SAMPLE TOO SHORT" or row["classification"] == "SAMPLE_TOO_SHORT" for row in report["matrix"])
    assert any("SAMPLE TOO SHORT" in d for d in report["disclaimers"])
    # no profitability language in classifications as claims
    assert "PROFITABLE" not in str(report)


def test_mtf_alignment(nifty_frames):
    from iqrp.app.backtesting.alpha_research.mtf import align_feature_to_execution
    from iqrp.app.backtesting.alpha_research.features import get_feature_registry

    f15 = nifty_frames["15m"]
    f5 = nifty_frames["5m"]
    feat, _ = get_feature_registry().compute(f15, "momentum", parameters={"lookback": 5})
    aligned = align_feature_to_execution(f15, feat, f5["timestamp"])
    assert len(aligned) == len(f5)


def test_signal_to_runner_path(nifty_frames, tmp_path):
    path = NIFTY["5m"]
    StrategyRegistry.register(LongShortMomentumStrategy, overwrite=True)
    cfg = BacktestRunConfig(
        backtest_id="alpha_research_pipeline_check",
        dataset_path=str(path),
        strategy_id="long_short_momentum",
        output_dir=str(tmp_path / "bt"),
        initial_capital=1_000_000,
        commission_bps=1.0,
        spread_bps=2.0,
        slippage_bps=2.0,
        seed=1,
        frequency="5m",
    )
    runner = BacktestRunner(cfg, strategy=LongShortMomentumStrategy(lookback=1, holding_bars=3, allow_short=True))
    runner.create()
    runner.validate()
    runner.prepare()
    result = runner.run()
    status = getattr(result.status, "value", result.status)
    assert str(status) == "COMPLETED"
    assert len(result.fills or []) >= 2
    recon = result.reconciliation if isinstance(result.reconciliation, dict) else {}
    if "ok" in recon:
        assert recon["ok"] is True
