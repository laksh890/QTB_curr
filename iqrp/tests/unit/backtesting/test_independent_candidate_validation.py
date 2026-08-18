"""Tests for independent candidate validation."""

from __future__ import annotations

from iqrp.app.backtesting.independent_validation.protocol import (
    DISCLAIMER,
    FROZEN_PRIMARY,
    NEGATIVE_CONTROL_ID,
    IndependentValidationConfig,
    classify_candidate,
    classify_holdout_duration,
)
from iqrp.app.backtesting.independent_validation.provenance import firewall_end_timestamp


def test_protocol_gates_and_live_false():
    cfg = IndependentValidationConfig()
    d = cfg.to_dict()
    assert d["live_ready"] is False
    assert "LIVE READY" in DISCLAIMER or "LIVE_READY" in DISCLAIMER
    assert len(FROZEN_PRIMARY) == 2
    assert NEGATIVE_CONTROL_ID.startswith("mdc_")


def test_duration_classification():
    assert classify_holdout_duration(1) == "INVALID_HOLDOUT"
    assert classify_holdout_duration(29) == "INVALID_HOLDOUT"
    assert classify_holdout_duration(30) == "INSUFFICIENT_HOLDOUT"
    assert classify_holdout_duration(179) == "INSUFFICIENT_HOLDOUT"
    assert classify_holdout_duration(180) == "DURATION_ADEQUATE"


def test_candidate_class_duration_dominates():
    # Even strong sharpe cannot escape INVALID_HOLDOUT
    assert (
        classify_candidate(
            duration_status="INVALID_HOLDOUT",
            net_sharpe=10.0,
            survives_base=True,
            survives_moderate=True,
            regime_ok=True,
            stat_ok=True,
        )
        == "INVALID_HOLDOUT"
    )


def test_firewall_after_p42():
    end = firewall_end_timestamp()
    assert "2026-07-31" in str(end)
