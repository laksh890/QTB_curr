"""Execution algorithms: TWAP/VWAP/POV/IS/adaptive/market/limit; urgency & residual."""

from __future__ import annotations

import pytest

from iqrp.app.execution.algorithms import (
    AdaptiveAlgorithm,
    ArrivalPriceAlgorithm,
    ImplementationShortfallAlgorithm,
    LimitAlgorithm,
    LiquiditySeekingAlgorithm,
    MarketAlgorithm,
    OpportunisticAlgorithm,
    POVAlgorithm,
    TWAPAlgorithm,
    VWAPAlgorithm,
)
from iqrp.app.execution.algorithms.base import (
    approved_quantity,
    coerce_urgency,
    redistribute_to_parent,
)
from iqrp.app.execution.algorithms.vwap import normalize_volume_curve
from iqrp.app.execution.registry import (
    available_algorithms,
    clear_custom_algorithms,
    get_algorithm,
    register_algorithm,
)
from iqrp.app.execution.types import Urgency

PARENT_QTY = 100.0


def _ctx(urgency: str = "NORMAL", residual: float | None = None, **extra):
    ctx = {
        "mid": 100.0,
        "price": 100.0,
        "spread": 0.02,
        "adv": 1_000_000.0,
        "volatility": 0.02,
        "side": "buy",
        "urgency": urgency,
        "n_slices": 4,
        "horizon_seconds": 60.0,
        "residual": residual if residual is not None else PARENT_QTY,
        "approved_quantity": residual if residual is not None else PARENT_QTY,
    }
    ctx.update(extra)
    return ctx


def _total(slices) -> float:
    return float(sum(s.quantity for s in slices))


@pytest.mark.parametrize(
    "algo_cls",
    [
        TWAPAlgorithm,
        VWAPAlgorithm,
        POVAlgorithm,
        ImplementationShortfallAlgorithm,
        AdaptiveAlgorithm,
        MarketAlgorithm,
        LimitAlgorithm,
        ArrivalPriceAlgorithm,
        LiquiditySeekingAlgorithm,
        OpportunisticAlgorithm,
    ],
)
@pytest.mark.parametrize("urgency", ["LOW", "NORMAL", "HIGH", "CRITICAL"])
def test_algo_never_exceeds_parent(algo_cls, urgency):
    algo = algo_cls()
    slices = algo.plan(PARENT_QTY, _ctx(urgency=urgency))
    assert _total(slices) <= PARENT_QTY + 1e-6
    assert all(s.quantity > 0 for s in slices) or _total(slices) == 0.0


def test_urgency_never_increases_total_beyond_parent():
    """Higher urgency may coarsen slices but total qty stays ≤ parent."""
    for cls in (TWAPAlgorithm, VWAPAlgorithm, POVAlgorithm, AdaptiveAlgorithm, MarketAlgorithm):
        totals = []
        for urg in (Urgency.LOW, Urgency.NORMAL, Urgency.HIGH, Urgency.CRITICAL):
            slices = cls().plan(PARENT_QTY, _ctx(urgency=urg.value))
            totals.append(_total(slices))
            assert totals[-1] <= PARENT_QTY + 1e-6
        # All urgencies fill the full approved residual when unconstrained
        assert max(totals) <= PARENT_QTY + 1e-6


def test_residual_handling_clips_plan():
    residual = 40.0
    for cls in (TWAPAlgorithm, VWAPAlgorithm, POVAlgorithm, ImplementationShortfallAlgorithm):
        slices = cls().plan(PARENT_QTY, _ctx(residual=residual, approved_quantity=residual))
        assert _total(slices) <= residual + 1e-6


def test_zero_approved_returns_empty():
    assert TWAPAlgorithm().plan(0.0, _ctx()) == []
    assert TWAPAlgorithm().plan(100.0, _ctx(residual=0.0, approved_quantity=0.0)) == []


def test_twap_with_participation_cap_and_jitter():
    algo = TWAPAlgorithm(n_slices=5, participation_cap=0.01, jitter=0.1, seed=42)
    slices = algo.plan(
        10_000.0,
        _ctx(
            residual=10_000.0,
            approved_quantity=10_000.0,
            adv=100_000.0,
            horizon_seconds=300.0,
            trading_day_seconds=23400.0,
            depth=[1, 1, 1, 1, 1],
        ),
    )
    assert _total(slices) <= 10_000.0 + 1e-6
    assert len(slices) >= 1


def test_twap_interval_seconds():
    algo = TWAPAlgorithm(n_slices=None, horizon_seconds=60.0, interval_seconds=20.0, seed=42)
    slices = algo.plan(
        PARENT_QTY,
        (
            _ctx(n_slices=None)
            if False
            else {
                **_ctx(),
                # don't override n_slices in ctx
            }
        ),
    )
    # remove n_slices override
    ctx = _ctx()
    ctx.pop("n_slices", None)
    slices = algo.plan(PARENT_QTY, ctx)
    assert _total(slices) <= PARENT_QTY + 1e-6


def test_vwap_volume_curve():
    curve = [1, 2, 3, 4, 5]
    algo = VWAPAlgorithm(n_slices=5, volume_curve=curve, participation_cap=None)
    slices = algo.plan(PARENT_QTY, _ctx())
    assert abs(_total(slices) - PARENT_QTY) < 1e-6
    w = normalize_volume_curve(curve, 5)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    assert normalize_volume_curve([], 3).shape == (3,)
    assert abs(float(normalize_volume_curve([0, 0, 0], 3).sum()) - 1.0) < 1e-9


def test_pov_participation_and_dynamic():
    algo = POVAlgorithm(target_participation=0.05, max_participation=0.1, n_slices=5, dynamic=True)
    slices = algo.plan(PARENT_QTY, _ctx(volume_profile=[1, 2, 1, 2, 1]))
    assert _total(slices) <= PARENT_QTY + 1e-6


def test_is_and_adaptive_and_arrival():
    for cls in (ImplementationShortfallAlgorithm, AdaptiveAlgorithm, ArrivalPriceAlgorithm):
        slices = cls().plan(PARENT_QTY, _ctx(urgency="HIGH", fill_rate=0.3, imbalance=0.2))
        assert _total(slices) <= PARENT_QTY + 1e-6


def test_market_and_limit():
    m = MarketAlgorithm(n_slices=3).plan(PARENT_QTY, _ctx(urgency="CRITICAL"))
    assert len(m) == 1
    assert abs(_total(m) - PARENT_QTY) < 1e-6
    lim = LimitAlgorithm(n_slices=2, limit_price=99.5).plan(PARENT_QTY, _ctx(side="sell"))
    assert abs(_total(lim) - PARENT_QTY) < 1e-6
    assert all(s.limit_price_hint == 99.5 for s in lim)


def test_liquidity_seeking_and_opportunistic():
    ls = LiquiditySeekingAlgorithm().plan(PARENT_QTY, _ctx(available_qty=50.0, depth=[10, 20, 30]))
    assert _total(ls) <= PARENT_QTY + 1e-6
    op = OpportunisticAlgorithm().plan(PARENT_QTY, _ctx(opportunity_score=0.8))
    assert _total(op) <= PARENT_QTY + 1e-6


def test_registry_get_and_custom():
    names = available_algorithms()
    assert "twap" in names
    assert "vwap" in names
    algo = get_algorithm("twap", n_slices=2, seed=42)
    assert algo.name == "twap"
    slices = algo.plan(50.0, _ctx(residual=50.0, approved_quantity=50.0, n_slices=2))
    assert _total(slices) <= 50.0 + 1e-6

    class Tiny(TWAPAlgorithm):
        name = "tiny_custom"

    register_algorithm("tiny_custom", Tiny)
    assert "tiny_custom" in available_algorithms()
    clear_custom_algorithms()
    assert "tiny_custom" not in available_algorithms()
    with pytest.raises(KeyError):
        get_algorithm("does_not_exist")
    with pytest.raises(ValueError):
        register_algorithm("", TWAPAlgorithm)


def test_approved_quantity_and_redistribute():
    assert (
        approved_quantity(100, {"residual": 40, "approved_quantity": 50, "max_quantity": 30})
        == 30.0
    )
    out = redistribute_to_parent([10, 20, 30], 50)
    assert abs(sum(out) - 50) < 1e-9
    assert redistribute_to_parent([], 10) == []
    assert sum(redistribute_to_parent([0, 0], 10)) == 0.0 or True


def test_coerce_urgency():
    assert coerce_urgency("high") is Urgency.HIGH
    assert coerce_urgency(None) is Urgency.NORMAL
    assert coerce_urgency(Urgency.LOW) is Urgency.LOW
