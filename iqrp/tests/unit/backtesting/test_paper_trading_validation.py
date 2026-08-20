"""Tests for Prompt 43 paper trading validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqrp.app.paper_trading.fill_model import AssumedFillModel
from iqrp.app.paper_trading.protocol import (
    DISCLAIMER,
    FROZEN_CANDIDATES,
    PaperTradingValidationConfig,
    classify_paper_status,
)
from iqrp.app.paper_trading.risk import KillSwitchState, PaperRiskLimits, check_risk
from iqrp.app.paper_trading.simulator import reconcile_session, run_sequential_paper


def test_protocol_no_live_ready():
    cfg = PaperTradingValidationConfig()
    d = cfg.to_dict()
    assert d["live_ready"] is False
    assert "LIVE_READY" in DISCLAIMER
    assert len(FROZEN_CANDIDATES) == 3


def test_fill_model_assumed_label():
    rng = np.random.default_rng(0)
    m = AssumedFillModel(
        {"commission_bps": 1, "half_spread_bps": 1, "slippage_bps": 1, "latency_bars": 1, "partial_fill_prob": 0, "reject_prob": 0, "variable_spread_bps": 0},
        rng=rng,
    )
    f = m.simulate(
        side="BUY",
        qty=1.0,
        mid=100.0,
        signal_ts="t0",
        order_ts="t0",
        fill_ts="t1",
        candidate_id="A",
    )
    assert f.cost_model_label == "ASSUMED_OHLCV_MICROSTRUCTURE"
    assert f.fill_price >= 100.0
    assert f.status == "FILLED"


def test_kill_switch_trips_on_recon():
    kill = KillSwitchState()
    limits = PaperRiskLimits()
    w, reasons = check_risk(
        limits=limits,
        kill=kill,
        target_weight=0.1,
        current_weight=0.0,
        equity=100.0,
        peak_equity=100.0,
        day_start_equity=100.0,
        recon_failed=True,
    )
    assert w == 0.0
    assert kill.halted
    assert "RECONCILIATION_FAILURE" in reasons


def test_sequential_no_future_and_recon():
    rng = np.random.default_rng(1)
    n = 80
    ts = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    px = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, n)))
    tw = np.sign(rng.normal(size=n)) * 0.1
    fill = AssumedFillModel(
        {"commission_bps": 1, "half_spread_bps": 1, "slippage_bps": 1, "latency_bars": 1, "partial_fill_prob": 0, "reject_prob": 0, "variable_spread_bps": 0},
        rng=rng,
    )
    out = run_sequential_paper(
        timestamps=ts.to_series().reset_index(drop=True),
        closes=px,
        target_weights=tw,
        fill_model=fill,
        limits=PaperRiskLimits(),
        initial_capital=100_000.0,
        latency_bars=1,
        candidate_label="test",
    )
    assert out["final_recon"]["ok"]
    assert out["lookahead_violations"] == 0
    assert out["n_bars"] == n


def test_classify_requires_gates():
    gates = {k: True for k in [
        "candidates_frozen", "no_lookahead", "sequential_ok", "execution_model_ok",
        "recon_zero_drift", "fills_to_positions_ok", "fees_accounted", "risk_limits_enforced",
        "kill_switches_ok", "failure_injection_ok", "no_2025_retune", "reproducible",
        "paper_pnl_positive_base", "survivors_exist",
    ]}
    assert classify_paper_status(gates) == "PAPER_TRADING_CANDIDATE"
    gates["recon_zero_drift"] = False
    assert classify_paper_status(gates) in {"PAPER_VALIDATION_WEAK", "PAPER_SIMULATION_OPERATIONAL"}
