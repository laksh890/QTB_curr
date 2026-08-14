"""Unit tests for Prompt 35 campaign helpers (no full BTC matrix)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.campaign import (
    _exclude_gap_contaminated,
    _gap_report,
    research_universe,
)
from iqrp.app.backtesting.alpha_research.types import (
    COST_SCENARIOS,
    ResearchStatus,
    bars_per_day,
    holding_clock_minutes,
    map_alpha_to_research_status,
)
from iqrp.app.backtesting.horizon.walk_forward import apply_purge_embargo, evaluate_oos
from iqrp.app.backtesting.scenarios.regime import classify_simple_regimes


def test_crypto_bars_per_day_and_clock():
    assert bars_per_day("5m", market_type="crypto") == 288.0
    assert holding_clock_minutes("15m", 4) == 60.0
    assert "BASE" in COST_SCENARIOS and "ADVERSE" in COST_SCENARIOS


def test_research_status_mapping():
    assert map_alpha_to_research_status("ROBUST_ALPHA") == ResearchStatus.CANDIDATE.value
    assert map_alpha_to_research_status("OOS_FAILURE") == ResearchStatus.OOS_FAILED.value
    assert map_alpha_to_research_status("COST_INEFFICIENT") == ResearchStatus.COST_INEFFICIENT.value


def test_purge_embargo_and_oos():
    g = np.random.default_rng(0).normal(0, 0.01, size=100)
    out = evaluate_oos(g, g, train_frac=0.5, validation_frac=0.25, purge_bars=2, embargo_bars=2)
    assert out["purge_bars"] == 2
    assert out["train"]["n"] <= 50
    parts = {"train": np.arange(50), "validation": np.arange(50, 75), "oos": np.arange(75, 100)}
    purged = apply_purge_embargo(parts, purge_bars=5, embargo_bars=3)
    assert purged["train"].size == 45
    assert purged["validation"][0] == 53


def test_regime_vectorized_large():
    r = np.random.default_rng(1).normal(0, 0.01, size=50_000)
    labels = classify_simple_regimes(r)
    assert labels.shape == (50_000,)
    assert len(set(labels.tolist())) >= 2


def test_gap_exclusion():
    ts = pd.date_range("2020-01-01", periods=20, freq="5min", tz="UTC").to_series()
    # introduce a gap
    ts.iloc[10] = ts.iloc[9] + pd.Timedelta(hours=2)
    frame = pd.DataFrame({"timestamp": ts.to_numpy()})
    rep = _gap_report(frame, "5m")
    assert rep["n_gaps"] >= 1
    keep = _exclude_gap_contaminated(frame["timestamp"], 3, "5m")
    assert keep.sum() < len(keep)


def test_research_universe_nonempty():
    u = research_universe()
    assert len(u["features"]) >= 7
    assert len(u["signals"]) >= 7
