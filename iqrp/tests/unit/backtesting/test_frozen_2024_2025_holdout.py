"""Tests for frozen 2024→2025 holdout validation."""

from __future__ import annotations

import pandas as pd

from iqrp.app.backtesting.frozen_2025_holdout.firewall import audit_firewall
from iqrp.app.backtesting.frozen_2025_holdout.protocol import (
    DISCLAIMER,
    EVIDENCE_IDS,
    HOLDOUT_START,
    RESEARCH_END,
    Frozen2025Config,
    classify_candidate,
)


def test_protocol_firewall_constants():
    cfg = Frozen2025Config()
    d = cfg.to_dict()
    assert d["live_ready"] is False
    assert "2024-12-31" in RESEARCH_END
    assert "2025-01-01" in HOLDOUT_START
    assert len(EVIDENCE_IDS) == 3
    assert "LIVE_READY" in DISCLAIMER


def test_firewall_detects_overlap():
    idx_r = pd.date_range("2024-12-01", periods=5, freq="h", tz="UTC")
    idx_h = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    research = {"1h": pd.DataFrame({"timestamp": idx_r, "close": 1.0})}
    holdout = {"1h": pd.DataFrame({"timestamp": idx_h, "close": 1.0})}
    concat = {
        "1h": pd.concat([research["1h"], holdout["1h"]], ignore_index=True),
    }
    audit = audit_firewall(research_frames=research, holdout_frames=holdout, concat_frames=concat)
    assert audit["status"] == "PASS"
    assert audit["hard_stop"] is False


def test_firewall_fails_on_contamination():
    idx = pd.date_range("2024-12-30", periods=10, freq="h", tz="UTC")  # crosses into 2025
    bad = {"1h": pd.DataFrame({"timestamp": idx, "close": 1.0})}
    audit = audit_firewall(research_frames=bad, holdout_frames=bad, concat_frames=bad)
    assert audit["status"] == "FAIL"
    assert audit["hard_stop"] is True


def test_classify_rejects_without_positive_net():
    row = {
        "complete_2025_holdout": True,
        "firewall_pass": True,
        "net_return": -0.1,
        "net_sharpe": -1.0,
        "survives_BASE": False,
        "survives_MODERATE": False,
        "max_drawdown": 0.1,
        "n_trades": 100,
        "causality_pass": True,
        "statistically_meaningful": False,
        "stable_through_2025": False,
        "reproducible": True,
        "not_single_period": False,
        "not_single_trade": True,
        "sharpe_not_inflated": True,
    }
    g = classify_candidate(row)
    assert g["status"] == "REJECTED"
