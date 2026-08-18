"""Focused tests for Prompt 42 final trading validation."""

from __future__ import annotations

import numpy as np

from iqrp.app.backtesting.final_validation.protocol import (
    DISCLAIMER,
    GATE_MIN_OOS_SHARPE,
    FinalValidationConfig,
    classify_behavior,
)
from iqrp.app.backtesting.final_validation.runner import (
    apply_profitability_gate,
    effective_sample_size,
    newey_west_se,
    walk_forward_folds,
)


def test_disclaimer_and_gates_frozen():
    cfg = FinalValidationConfig()
    d = cfg.to_dict()
    assert "PROFITABILITY_EVIDENCE" in DISCLAIMER
    assert "LIVE_READY" in DISCLAIMER
    assert GATE_MIN_OOS_SHARPE == 0.0
    assert "positive_oos_sharpe" in d["profitability_gate"]["required_all"]
    assert d["run_predeclared_grid"] is False


def test_behavior_classifier():
    assert classify_behavior(0.01) == "BUY_AND_HOLD"
    assert classify_behavior(0.2) == "LOW_FREQUENCY"
    assert classify_behavior(1.0) == "SWING"
    assert classify_behavior(5.0) == "INTRADAY"
    assert classify_behavior(20.0) == "HIGH_FREQUENCY_RESEARCH"
    assert classify_behavior(60.0) == "OVERTRADING"


def test_effective_sample_size_and_hac():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    # induce AR(1)
    y = np.zeros(500)
    for i in range(1, 500):
        y[i] = 0.5 * y[i - 1] + x[i]
    n_eff = effective_sample_size(500, 0.5)
    assert n_eff < 500
    se = newey_west_se(y)
    assert np.isfinite(se) and se > 0


def test_walk_forward_folds_chronological():
    folds = walk_forward_folds(1000, n_folds=3)
    assert folds
    for f in folds:
        assert f["train"].stop <= f["test"].start or f["train"].stop == f["test"].start
        assert f["test"].stop <= f["final_oos"].start
        assert f["final_oos"].stop == 1000


def test_profitability_gate_rejects_incomplete():
    row = {
        "oos_net_return": 0.1,
        "oos_net_sharpe": 0.5,
        "expectancy": 0.01,
        "survives_BASE": True,
        "survives_MODERATE": False,
        "adverse_net_sharpe": -0.5,
        "leakage_ok": True,
        "recon_ok": True,
        "execution_timing_ok": True,
        "walk_forward_ok": True,
        "not_tiny_window": True,
        "oos_max_dd": 0.1,
        "acceptable_turnover": True,
        "n_trades_oos": 100,
        "no_oos_contamination_in_selection": True,
        "perturb_survival": 0.8,
        "regime_ok": True,
        "reproducible": True,
    }
    g = apply_profitability_gate(row)
    assert g["status"] != "PROFITABILITY_EVIDENCE"
    assert "survives_MODERATE" in g["failed_checks"]


def test_data_provenance_resolve_keys():
    from iqrp.app.backtesting.final_validation.data_provenance import resolve_dataset_keys

    resolved = resolve_dataset_keys("dataset_registry.json")
    assert "dataset_keys" in resolved
    assert "1m" in resolved["dataset_keys"]
