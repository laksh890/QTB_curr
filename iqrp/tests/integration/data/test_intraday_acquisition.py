"""Integration tests: provider → canonical → registry → backtest (fixtures + optional live)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.data.dataset_registry import DatasetRegistry
from iqrp.app.backtesting.data.schema import normalize_frame
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.runner import BacktestRunner
from iqrp.app.backtesting.strategy.long_short_momentum import LongShortMomentumStrategy
from iqrp.app.backtesting.strategy.registry import StrategyRegistry
from iqrp.app.data.historical.calendar import frequency_to_seconds, nse_equity_calendar
from iqrp.app.data.historical.intraday_validation import build_intraday_quality_report
from iqrp.app.data.historical.pipeline import AcquisitionPipeline
from iqrp.app.data.historical.provider import ProviderRequest
from iqrp.app.data.historical.resampling import resample_session_aware
from iqrp.app.data.historical.yahoo_finance import YahooFinanceHistoricalProvider


def _make_intraday_fixture(tmp_path: Path, n_days: int = 2) -> Path:
    cal = nse_equity_calendar()
    d = date(2026, 8, 10)
    rows = []
    px = 24000.0
    rng = np.random.default_rng(7)
    days_done = 0
    while days_done < n_days:
        if cal.is_trading_day(d):
            for ts in cal.expected_bar_timestamps(d, 60):
                shock = float(rng.normal(0, 2.0))
                o, c = px, px + shock
                rows.append(
                    {
                        "timestamp": pd.Timestamp(ts).tz_convert("UTC"),
                        "instrument": "NIFTY50",
                        "open": o,
                        "high": max(o, c) + 0.5,
                        "low": min(o, c) - 0.5,
                        "close": c,
                        "volume": float(rng.integers(100, 5000)),
                    }
                )
                px = c
            days_done += 1
        d = date.fromordinal(d.toordinal() + 1)
    frame = normalize_frame(pd.DataFrame(rows))
    path = tmp_path / "fixture_1m.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_raw_to_canonical_to_registry(tmp_path: Path):
    path = _make_intraday_fixture(tmp_path)
    frame = normalize_frame(pd.read_parquet(path))
    quality = build_intraday_quality_report(frame, frequency="1m", dataset_id="fixture_1m")
    assert quality["ok"] is True
    derived, prov = resample_session_aware(
        frame, source_frequency="1m", derived_frequency="5m", source_dataset_id="fixture@1"
    )
    dpath = tmp_path / "fixture_5m.parquet"
    derived.to_parquet(dpath, index=False)
    reg = DatasetRegistry(tmp_path / "reg.json")
    from iqrp.app.data.historical.registry_ops import register_immutable
    from iqrp.app.data.historical.provenance import DatasetProvenance

    prov.checksum = "x"
    register_immutable(
        reg,
        path=dpath,
        dataset_id="nifty50_intraday_5m",
        version="1.0.0",
        source="derived:fixture",
        frame=derived,
        provenance=prov,
        quality_status="PASS",
    )
    rec = reg.require("nifty50_intraday_5m", "1.0.0")
    assert rec.extra.get("frequency_kind") == "DERIVED"
    assert rec.checksum


def test_canonical_to_backtest_multiple_trades(tmp_path: Path):
    path = _make_intraday_fixture(tmp_path, n_days=3)
    StrategyRegistry.register(LongShortMomentumStrategy, overwrite=True)
    out = tmp_path / "bt"
    cfg = BacktestRunConfig(
        backtest_id="intraday_pipeline_validation",
        dataset_path=str(path),
        strategy_id="long_short_momentum",
        strategy_version="1.0.0",
        output_dir=str(out),
        initial_capital=1_000_000.0,
        commission_bps=1.0,
        spread_bps=2.0,
        slippage_bps=2.0,
        seed=42,
        frequency="1m",
    )
    runner = BacktestRunner(
        cfg,
        strategy=LongShortMomentumStrategy(lookback=1, holding_bars=5, allow_short=True),
    )
    runner.create()
    runner.validate()
    runner.prepare()
    result = runner.run()
    status = getattr(result.status, "value", result.status)
    assert str(status) == "COMPLETED"
    trades = list(result.trades or [])
    recon = result.reconciliation if isinstance(result.reconciliation, dict) else {}
    # Prefer explicit multi-trade: fills or trades or orders
    n_fills = len(result.fills or [])
    n_trades = len(trades)
    n_orders = len(result.orders or [])
    assert n_fills >= 2 or n_trades >= 2 or n_orders >= 2
    # Reconciliation should not be explicitly failed when present
    if "ok" in recon:
        assert recon["ok"] is True


def test_yahoo_provider_live_optional():
    """Live Yahoo probe — skipped if network/provider fails."""
    try:
        prov = YahooFinanceHistoricalProvider()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=5)
        resp = prov.download(
            ProviderRequest(
                instrument="NIFTY50",
                start=start,
                end=end,
                frequency="1m",
                adjustment_policy="unadjusted",
            )
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"yahoo unavailable: {exc}")
    assert len(resp.frame) > 0
    assert resp.data_class == "DEVELOPMENT DATA"
    assert resp.license_status == "UNKNOWN"
    assert resp.frame["timestamp"].dt.tz is not None
