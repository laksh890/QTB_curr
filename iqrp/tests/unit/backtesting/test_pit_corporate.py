"""Point-in-time validation and corporate actions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from iqrp.app.backtesting.corporate_actions import (
    CorporateAction,
    CorporateActionType,
    PositionState,
    actions_asof,
    adjust_price_for_dividend,
    adjust_price_for_split,
    adjust_quantity_for_split,
    apply_corporate_actions,
    build_action,
    cumulative_split_factor,
)
from iqrp.app.backtesting.pit import (
    LeakageReport,
    LookaheadViolation,
    assert_no_lookahead,
    available_asof,
    detect_leakage,
    filter_frame_asof,
    filter_universe_asof,
)


def _dt(day: int, month: int = 1) -> datetime:
    return datetime(2020, month, day, tzinfo=UTC)


# --------------------------------------------------------------------------- PIT
def test_assert_no_lookahead_ok_and_fail() -> None:
    assert_no_lookahead(5, 10)
    assert_no_lookahead(10, 10, allow_equal=True)
    with pytest.raises(LookaheadViolation):
        assert_no_lookahead(11, 10, context="close")
    with pytest.raises(LookaheadViolation):
        assert_no_lookahead(10, 10, allow_equal=False)
    with pytest.raises(LookaheadViolation):
        assert_no_lookahead(datetime(2020, 1, 2), datetime(2020, 1, 1))  # naive? need aware
    # naive datetimes rejected
    with pytest.raises(LookaheadViolation):
        assert_no_lookahead(datetime(2020, 1, 1), 5)


def test_assert_no_lookahead_aware() -> None:
    assert_no_lookahead(_dt(1), _dt(2))
    with pytest.raises(LookaheadViolation):
        assert_no_lookahead(_dt(3), _dt(2), context="px")


def test_detect_leakage() -> None:
    report = detect_leakage([0, 1, 2], [0, 1, 3], timestamps=[1, 2, 3])
    assert report.has_leakage
    assert bool(report) is True
    assert report.n_violations >= 1
    assert "sample" in repr(report)

    clean = detect_leakage([0, 1, 2], [0, 1, 2])
    assert not clean.has_leakage

    horiz = detect_leakage([0, 1, 2], [0, 1, 2], max_label_horizon=0)
    # equal indices → horizon 0, ok when max=0; use larger gap
    horiz2 = detect_leakage([0, 0, 0], [0, 1, 2], max_label_horizon=0)
    assert horiz2.has_leakage

    with pytest.raises(ValueError):
        detect_leakage([0, 1], [0])

    past = detect_leakage([0, 5], [0, 5], timestamps=[1, 2, 3])
    assert past.has_leakage


def test_filter_universe_asof_forms(membership) -> None:
    assert set(filter_universe_asof(membership, 50)) == {"AAA", "BBB", "CCC"}
    assert "DDD" not in filter_universe_asof(membership, 50)
    assert "AAA" not in filter_universe_asof(membership, 80)

    mapping = {"X": {"start": 0, "end": 10}, "Y": {"start": 5, "end": None}}
    assert filter_universe_asof(mapping, 5) == ["X", "Y"]
    assert filter_universe_asof(mapping, 10) == ["Y"]

    rows = [
        {"symbol": "A", "start": 0, "end": 5},
        {"symbol": "B", "start": 3, "end": None},
    ]
    assert filter_universe_asof(rows, 4) == ["A", "B"]

    with pytest.raises(LookaheadViolation):
        filter_universe_asof({"Z": "bad"}, 1)


def test_filter_frame_and_available_asof() -> None:
    ts = [0, 1, 2, 3, 4]
    assert filter_frame_asof(ts, 2) == [0, 1, 2]
    items = [("a", 1), ("b", 3), ("c", 2)]
    assert available_asof(items, 2) == ["a", "c"]


# --------------------------------------------------------------------------- corporate
def test_actions_asof_filters_future() -> None:
    acts = [
        build_action("SPLIT", "AAA", _dt(5), ratio=2.0),
        build_action("DIVIDEND", "AAA", _dt(10), amount=0.5),
        build_action("MERGER", "BBB", _dt(20), new_symbol="CCC"),
    ]
    got = actions_asof(acts, _dt(10))
    assert len(got) == 2
    assert all(a.ex_date <= _dt(10) for a in got)


def test_split_dividend_helpers() -> None:
    assert adjust_price_for_split(100.0, 2.0) == 50.0
    assert adjust_quantity_for_split(10.0, 2.0) == 20.0
    with pytest.raises(ValueError):
        adjust_price_for_split(100.0, 0)
    assert adjust_price_for_dividend(100.0, 2.0, method="subtract") == 98.0
    factored = adjust_price_for_dividend(100.0, 2.0, method="factor")
    assert factored == pytest.approx(98.0)
    with pytest.raises(ValueError):
        adjust_price_for_dividend(0.0, 1.0, method="factor")
    with pytest.raises(ValueError):
        adjust_price_for_dividend(100.0, 1.0, method="weird")


def test_apply_corporate_actions_all_types() -> None:
    asof = _dt(30)
    positions = {
        "AAA": PositionState("AAA", quantity=100.0, cost_basis=50.0),
        "BBB": 50.0,
        "OLD": PositionState("OLD", quantity=20.0),
        "DEAD": PositionState("DEAD", quantity=10.0),
        "RENAME": PositionState("RENAME", quantity=5.0),
    }
    actions = [
        build_action("SPLIT", "AAA", _dt(5), ratio=2.0, action_id="s1"),
        build_action("DIVIDEND", "AAA", _dt(6), amount=1.0),
        build_action("MERGER", "BBB", _dt(7), new_symbol="CCC", exchange_ratio=1.5),
        build_action("DELISTING", "DEAD", _dt(8), liquidation_price=10.0),
        build_action("SYMBOL_CHANGE", "RENAME", _dt(9), new_symbol="NEW"),
        build_action("OTHER", "AAA", _dt(10), note="noop"),
        # future — must not apply
        build_action("SPLIT", "AAA", _dt(15, month=2), ratio=3.0),
    ]
    # fix action_id on build — build_action uses kwargs as payload; set via CorporateAction
    actions[0] = CorporateAction(
        CorporateActionType.SPLIT, "AAA", _dt(5), {"ratio": 2.0}, action_id="s1"
    )
    result = apply_corporate_actions(positions, actions, asof=asof, cash=0.0)
    assert "AAA" in result.positions
    assert result.positions["AAA"].quantity == pytest.approx(200.0)
    assert result.cash_delta > 0
    assert "CCC" in result.positions
    assert "BBB" not in result.positions
    assert "DEAD" not in result.positions
    assert "NEW" in result.positions
    assert "RENAME" not in result.positions
    assert any(a.action_type == CorporateActionType.SPLIT for a in result.applied)


def test_delisting_without_liq_and_skip_missing() -> None:
    asof = _dt(10)
    actions = [
        build_action("DELISTING", "X", _dt(5)),
        build_action("DIVIDEND", "MISSING", _dt(5), amount=1.0),
        build_action("MERGER", "MISSING", _dt(5), new_symbol="Y"),
        build_action("SYMBOL_CHANGE", "MISSING", _dt(5), new_symbol="Z"),
    ]
    result = apply_corporate_actions({"X": 8.0}, actions, asof=asof)
    assert "X" not in result.positions
    assert result.notes


def test_corporate_action_naive_ex_date_rejected() -> None:
    with pytest.raises(ValueError):
        CorporateAction(CorporateActionType.SPLIT, "A", datetime(2020, 1, 1), {"ratio": 2})


def test_cumulative_split_factor() -> None:
    acts = [
        build_action("SPLIT", "AAA", _dt(5), ratio=2.0),
        build_action("SPLIT", "AAA", _dt(15), ratio=3.0),
        build_action("SPLIT", "BBB", _dt(5), ratio=2.0),
    ]
    assert cumulative_split_factor(acts, "AAA", asof=_dt(10)) == pytest.approx(2.0)
    assert cumulative_split_factor(acts, "AAA", asof=_dt(20)) == pytest.approx(6.0)


def test_leakage_report_repr_clean() -> None:
    r = LeakageReport(False, 3, 0)
    assert "has_leakage=False" in repr(r)
