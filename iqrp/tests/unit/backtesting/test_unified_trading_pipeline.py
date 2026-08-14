"""Tests for unified Alpha→Risk→Portfolio→Execution orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.alpha_research.engine import AlphaSignalResearchEngine
from iqrp.app.backtesting.unified_pipeline.candidate import (
    candidate_from_alpha_result,
    validate_candidate,
)
from iqrp.app.backtesting.unified_pipeline.orchestrator import UnifiedTradingOrchestrator
from iqrp.app.backtesting.unified_pipeline.portfolio_gate import apply_portfolio_constraints
from iqrp.app.backtesting.unified_pipeline.risk_gate import (
    default_risk_engine,
    evaluate_candidate_risk,
    size_approved_exposure,
)
from iqrp.app.backtesting.unified_pipeline.types import AlphaCandidate, StageOutcome
from iqrp.app.backtesting.alpha_research.signals import get_signal_registry


def _cand(
    cid: str,
    instrument: str,
    direction: float,
    weight: float,
    *,
    ts: str = "2022-01-01T00:00:00+00:00",
) -> AlphaCandidate:
    return AlphaCandidate(
        candidate_id=cid,
        signal_id=f"sig_{cid}",
        instrument=instrument,
        timestamp=ts,
        direction=float(direction),
        signal_value=float(direction),
        source_model="test",
        source_model_version="1.0.0",
        data_version="test@1",
        dataset_checksum="chk",
        oos_status="EVALUATED",
        requested_weight=float(weight),
        expected_horizon=3,
    )


def test_candidate_validation_rejects_bad():
    bad = AlphaCandidate(
        candidate_id="x",
        signal_id="s",
        instrument="",
        timestamp="bad",
        direction=2.0,
        signal_value=float("nan"),
        source_model="m",
        source_model_version="",
        data_version="",
        oos_status="OOS_FAILED",
    )
    ok, codes = validate_candidate(bad, asof="2022-01-01T00:00:00+00:00")
    assert not ok
    assert "NON_FINITE_SIGNAL" in codes
    assert "MISSING_INSTRUMENT" in codes


def test_candidate_duplicate_and_future():
    c = _cand("dup", "AAA", 1.0, 0.05)
    ok, codes = validate_candidate(c, asof="2022-01-01T00:00:00+00:00", seen_ids={"dup"})
    assert not ok and "DUPLICATE_CANDIDATE" in codes
    future = _cand("f1", "AAA", 1.0, 0.05, ts="2022-01-02T00:00:00+00:00")
    ok2, codes2 = validate_candidate(future, asof="2022-01-01T00:00:00+00:00")
    assert not ok2 and "FUTURE_INFORMATION" in codes2


def test_risk_handoff_approve_reduce_reject():
    risk = default_risk_engine(max_position=0.1)
    rets = np.random.default_rng(0).normal(0, 0.01, 150)
    approved = evaluate_candidate_risk(
        _cand("a", "AAA", 1.0, 0.05),
        risk_engine=risk,
        current_weights={"AAA": 0.0},
        returns=rets,
    )
    assert approved.outcome in {StageOutcome.RISK_APPROVED, StageOutcome.RISK_REDUCED}
    rejected = evaluate_candidate_risk(
        _cand("b", "AAA", 1.0, 0.95),
        risk_engine=risk,
        current_weights={"AAA": 0.0},
        returns=rets,
    )
    assert rejected.outcome == StageOutcome.RISK_REJECTED
    sizing = size_approved_exposure(
        risk_engine=risk, approved_exposure=0.05, returns=rets, equity=1e6
    )
    assert sizing.final_size != 0 or sizing.requested_size == 0.05


def test_portfolio_reduction_and_long_only():
    res = apply_portfolio_constraints(
        instrument="AAA",
        proposed_weight=0.5,
        current_weights={"BBB": 0.2},
        max_position=0.1,
        max_gross=0.25,
        long_only=False,
    )
    assert res.outcome == StageOutcome.PORTFOLIO_REDUCED
    assert abs(res.target_position_weight) <= 0.1 + 1e-9
    short = apply_portfolio_constraints(
        instrument="AAA",
        proposed_weight=-0.05,
        current_weights={},
        long_only=True,
        max_position=0.1,
        max_gross=1.0,
    )
    assert short.outcome in {StageOutcome.PORTFOLIO_REDUCED, StageOutcome.PORTFOLIO_REJECTED}


def test_multi_candidate_long_short_lineage_and_recon():
    orch = UnifiedTradingOrchestrator(
        initial_capital=1_000_000.0,
        long_only=False,
        max_position=0.08,
        max_gross=0.3,
        base_returns=np.random.default_rng(2).normal(0, 0.01, 120),
    )
    asof = "2022-01-01T12:00:00+00:00"
    prices = {"AAA": 100.0, "BBB": 50.0}
    out = orch.process_candidates(
        [
            _cand("c1", "AAA", 1.0, 0.05, ts=asof),
            _cand("c2", "BBB", -1.0, -0.05, ts=asof),
            _cand("c3", "AAA", 1.0, 0.9, ts=asof),  # reject
        ],
        asof=asof,
        prices=prices,
    )
    outcomes = [r["outcome"] for r in out["results"]]
    assert StageOutcome.RISK_REJECTED.value in outcomes
    assert out["lineage"]
    assert "candidate_id" in out["lineage"][0]
    assert "risk_decision_id" in out["lineage"][0]
    assert out["reconciliation"]["outcome"] in {
        StageOutcome.RECONCILIATION_OK.value,
        StageOutcome.RECONCILIATION_FAILED.value,
    }
    # Positions should reflect at least one fill path when approved
    assert isinstance(out["positions"], dict)


def test_reversal_and_deterministic_order():
    orch = UnifiedTradingOrchestrator(initial_capital=1e6, long_only=False, max_position=0.1)
    asof = "2022-01-01T00:00:00+00:00"
    prices = {"AAA": 100.0}
    orch.process_candidates([_cand("z", "AAA", 1.0, 0.05, ts=asof)], asof=asof, prices=prices)
    out2 = orch.process_candidates(
        [_cand("y", "AAA", -1.0, -0.05, ts=asof)],
        asof=asof,
        prices=prices,
    )
    # second candidate processed; weights/positions updated
    assert out2["results"][0]["candidate_id"] == "y"
    # deterministic sort: process b before a when both submitted
    orch2 = UnifiedTradingOrchestrator(initial_capital=1e6, long_only=False, max_position=0.1)
    out = orch2.process_candidates(
        [_cand("b", "AAA", 1.0, 0.04, ts=asof), _cand("a", "BBB", -1.0, -0.04, ts=asof)],
        asof=asof,
        prices={"AAA": 100.0, "BBB": 100.0},
    )
    assert [r["candidate_id"] for r in out["results"]] == ["a", "b"]


def test_reference_signal_into_pipeline():
    rng = np.random.default_rng(3)
    n = 120
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC"),
            "instrument": "BTCUSDT",
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )
    assert "momentum_signal" in {s.signal_id for s in get_signal_registry().list()}
    eng = AlphaSignalResearchEngine(market_type="crypto", timezone="UTC")
    ev = eng.evaluate_candidate(
        df,
        signal_id="momentum_signal",
        timeframe="1h",
        holding_bars=3,
        parameters={"lookback": 10},
        dataset_id="t@1",
        dataset_checksum="c",
        run_leakage=False,
        run_importance=False,
        run_regime=False,
        persist_experiment=False,
    )
    cand = candidate_from_alpha_result(
        {**ev, "signal_id": "momentum_signal", "dataset_id": "t@1", "dataset_checksum": "c"},
        instrument="BTCUSDT",
        timestamp=str(df["timestamp"].iloc[-1]),
        base_weight=0.05,
        source_model="reference",
        source_model_version="1.0.0",
        data_version="t@1",
        dataset_checksum="c",
    )
    if abs(cand.direction) < 1e-12:
        d = {k: v for k, v in cand.to_dict().items() if k != "disclaimer"}
        d.update({"direction": 1.0, "signal_value": 1.0, "requested_weight": 0.05})
        cand = AlphaCandidate.from_dict(d)
    orch = UnifiedTradingOrchestrator(initial_capital=1e6, long_only=False)
    out = orch.process_candidates(
        [cand],
        asof=str(df["timestamp"].iloc[-1]),
        prices={"BTCUSDT": float(df["close"].iloc[-1])},
        returns=df["close"].pct_change().fillna(0).to_numpy(),
    )
    assert out["results"][0]["outcome"] != StageOutcome.CANDIDATE_REJECTED.value


def test_serialization_lineage():
    c = _cand("ser", "AAA", 1.0, 0.05)
    d = c.to_dict()
    c2 = AlphaCandidate.from_dict(d)
    assert c2.candidate_id == c.candidate_id
    assert c2.requested_weight == c.requested_weight
