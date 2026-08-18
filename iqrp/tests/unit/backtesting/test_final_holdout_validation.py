"""Tests for independent final holdout validation."""

from __future__ import annotations

from iqrp.app.backtesting.final_holdout.freeze import definition_checksum, freeze_candidates
from iqrp.app.backtesting.final_holdout.protocol import (
    DISCLAIMER,
    FROZEN_CANDIDATE_IDS,
    FinalHoldoutConfig,
    classify_degradation,
)
from iqrp.app.backtesting.final_holdout.provenance import p42_research_windows


def test_disclaimer_and_live_false():
    cfg = FinalHoldoutConfig()
    d = cfg.to_dict()
    assert d["live_ready"] is False
    assert "LIVE READY" in DISCLAIMER or "LIVE_READY" in DISCLAIMER
    assert len(FROZEN_CANDIDATE_IDS) == 3


def test_degradation_classifier():
    assert classify_degradation(2.0, 1.8) == "STABLE"
    assert classify_degradation(2.0, 0.8) == "MODERATE_DEGRADATION"
    assert classify_degradation(2.0, 0.1) == "SEVERE_DEGRADATION"
    assert classify_degradation(2.0, -0.1) == "FAILED_REPLICATION"


def test_freeze_matches_prompt42():
    freeze = freeze_candidates()
    assert freeze["status"] == "PASS"
    assert freeze["n_frozen"] == 3
    assert freeze["all_definitions_match"]
    # checksum deterministic
    c0 = freeze["candidates"][0]
    assert definition_checksum(c0["definition"]) == c0["definition_checksum"]


def test_p42_windows_end_at_registered_series():
    w = p42_research_windows()
    assert w["latest_p42_timestamp"] is not None
    assert "2026-07-31" in w["latest_p42_timestamp"]
