"""Unit tests for alpha research engine primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.alpha_research.analytics import (
    decay_curve,
    evaluate_cost_aware,
    positions_from_signal,
    timeseries_ic_report,
)
from iqrp.app.backtesting.alpha_research.features import get_feature_registry
from iqrp.app.backtesting.alpha_research.leakage import LeakageError, run_leakage_suite
from iqrp.app.backtesting.alpha_research.normalize import causal_rolling_zscore
from iqrp.app.backtesting.alpha_research.ranking import classify_alpha, compute_alpha_research_score
from iqrp.app.backtesting.alpha_research.signals import apply_holding, get_signal_registry
from iqrp.app.backtesting.alpha_research.types import AlphaClassification


def _frame(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=n))
    ts = pd.date_range("2026-08-10 03:45", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "instrument": "NIFTY50",
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.integers(100, 1000, size=n).astype(float),
        }
    )


def test_feature_causality_and_registry():
    reg = get_feature_registry()
    ids = {s.feature_id for s in reg.list()}
    assert "momentum" in ids and "RSI" in ids
    df = _frame()
    series, meta = reg.compute(df, "momentum", parameters={"lookback": 5})
    assert meta["causal"] is True
    assert series.iloc[:5].isna().all()


def test_normalization_causal():
    s = pd.Series(np.arange(30, dtype=float))
    z = causal_rolling_zscore(s, window=5)
    s2 = s.copy()
    s2.iloc[-1] = 1e9
    z2 = causal_rolling_zscore(s2, window=5)
    assert np.allclose(z.iloc[:-2], z2.iloc[:-2], equal_nan=True)


def test_leakage_suite_passes_and_rejects_future_col():
    df = _frame()
    reg = get_feature_registry()
    feat, _ = reg.compute(df, "returns", parameters={"lookback": 1})
    out = run_leakage_suite(
        df, feat, lookback=1, compute_fn=lambda f: reg.compute(f, "returns", parameters={"lookback": 1})[0]
    )
    assert out["ok"]
    bad = df.copy()
    bad["future_ret"] = bad["close"].shift(-1)
    with pytest.raises(LeakageError):
        run_leakage_suite(bad, feat, lookback=1)


def test_signal_generation_long_short():
    reg = get_signal_registry()
    df = _frame(100)
    sig, meta, _ = reg.generate(df, "momentum_signal", parameters={"lookback": 5, "holding_bars": 3})
    assert set(np.unique(sig.dropna())) <= {-1.0, 0.0, 1.0}
    pos = apply_holding(sig.fillna(0), 3)
    assert (pos.abs() <= 1).all()


def test_mean_reversion_and_breakout():
    df = _frame(120, seed=2)
    sreg = get_signal_registry()
    a, _, _ = sreg.generate(df, "mean_reversion_signal", parameters={"lookback": 10})
    b, _, _ = sreg.generate(df, "breakout_signal", parameters={"lookback": 10})
    assert a.notna().sum() > 0 and b.notna().sum() > 0


def test_ic_and_decay():
    df = _frame(100)
    sig = np.sign(df["close"].pct_change().shift(1).fillna(0))
    ic = timeseries_ic_report(sig, df["close"])
    assert ic["not_cross_sectional_ic"] is True
    dec = decay_curve(sig, df["close"])
    assert "peak_predictive_horizon_bars" in dec


def test_cost_and_turnover():
    df = _frame(80)
    pos = positions_from_signal(pd.Series(np.sign(np.sin(np.arange(80)))), 2)
    ev = evaluate_cost_aware(pos, df["close"].pct_change().fillna(0), periods_per_year=252 * 75)
    assert "transaction_costs" in ev
    assert "alpha_survives_costs" in ev or "alpha_collapses_after_costs" in ev


def test_alpha_score_and_sample_too_short():
    scored = compute_alpha_research_score(
        {"net_sharpe": 1.0, "expectancy": 0.01, "oos_sharpe": 0.5, "mean_ic": 0.05, "max_drawdown": 0.1}
    )
    assert 0 <= scored["score"] <= 1
    cls, _ = classify_alpha({"net_sharpe": 1.0, "trade_count": 50, "alpha_survives_costs": True}, n_sessions=6)
    assert cls == AlphaClassification.SAMPLE_TOO_SHORT


def test_experiment_reproducibility(tmp_path):
    from iqrp.app.backtesting.alpha_research.experiments import ExperimentRegistry, ExperimentSpec, now_iso

    reg = ExperimentRegistry(tmp_path / "exp.json")
    spec = ExperimentSpec(
        experiment_id="e1",
        timestamp=now_iso(),
        dataset_id="x",
        dataset_checksum="abc",
        feature_versions={"momentum": "1.0.0"},
        signal_id="momentum_signal",
        signal_version="1.0.0",
        parameters={"lookback": 10},
        timeframe="5m",
        holding_period=5,
        cost_model={"commission_bps": 1},
        result_checksum=ExperimentRegistry.result_checksum({"a": 1}),
    )
    reg.register(spec)
    reg2 = ExperimentRegistry(tmp_path / "exp.json")
    assert len(reg2.list()) == 1
