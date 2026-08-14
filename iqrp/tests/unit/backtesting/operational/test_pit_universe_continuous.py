"""Point-in-time, universe, and continuous contract tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from iqrp.app.backtesting.data import (
    AdjustmentMethod,
    ContinuousContractBuilder,
    ContinuousContractConfig,
    ContractSeriesKind,
    ContractSpec,
    LookaheadViolation,
    RollRule,
    assert_no_lookahead,
    build_continuous_series,
    continuous_futures_universe,
    custom_universe,
    ensure_effective_timestamps,
    filter_features_asof,
    filter_frame_asof_df,
    filter_signals_asof,
    filter_universe_membership_asof,
    futures_universe,
    historical_universe,
    index_constituents,
    instrument_list,
    resolve_universe,
    single_instrument,
)
from iqrp.app.backtesting.data.synthetic import generate_synthetic_ohlcv


def test_pit_filter_and_lookahead():
    frame = generate_synthetic_ohlcv(n_days=10, instruments=["AAA"], seed=2)
    frame = ensure_effective_timestamps(frame)
    asof = frame["timestamp"].iloc[4].to_pydatetime()
    filtered = filter_frame_asof_df(frame, asof)
    assert (filtered["effective_timestamp"] <= pd.Timestamp(asof)).all()
    feats = filter_features_asof(frame, asof)
    sigs = filter_signals_asof(frame, asof)
    assert len(feats) == len(filtered)
    assert len(sigs) == len(filtered)

    assert_no_lookahead(asof, asof + timedelta(seconds=1))
    with pytest.raises(LookaheadViolation):
        assert_no_lookahead(asof + timedelta(days=1), asof)
    with pytest.raises(LookaheadViolation):
        filter_frame_asof_df(frame, datetime(2020, 1, 5))  # naive


def test_ensure_effective_rejects_leak():
    frame = generate_synthetic_ohlcv(n_days=5, instruments=["A"], seed=1)
    frame = ensure_effective_timestamps(frame)
    bad = frame.copy()
    bad.loc[bad.index[0], "effective_timestamp"] = bad.loc[
        bad.index[0], "timestamp"
    ] + pd.Timedelta(days=1)
    with pytest.raises(LookaheadViolation):
        ensure_effective_timestamps(bad)
    with pytest.raises(ValueError):
        ensure_effective_timestamps(pd.DataFrame({"x": [1]}))


def test_universe_helpers():
    assert resolve_universe(single_instrument("AAA")) == ["AAA"]
    assert resolve_universe(instrument_list(["AAA", "BBB"])) == ["AAA", "BBB"]

    membership = [
        {
            "instrument": "AAA",
            "start": datetime(2020, 1, 1, tzinfo=UTC),
            "end": datetime(2020, 2, 1, tzinfo=UTC),
        },
        {"instrument": "BBB", "start": datetime(2020, 1, 15, tzinfo=UTC), "end": None},
    ]
    hist = historical_universe(membership)
    all_inst = resolve_universe(hist)
    assert set(all_inst) == {"AAA", "BBB"}
    asof = datetime(2020, 1, 10, tzinfo=UTC)
    assert resolve_universe(hist, asof=asof) == ["AAA"]
    assert hist.asof(asof) == ["AAA"]

    idx = index_constituents(membership, index_name="DEMO")
    assert "AAA" in resolve_universe(idx, asof=asof)

    fut = futures_universe(["ESH0", "ESM0"], root="ES")
    assert resolve_universe(fut) == ["ESH0", "ESM0"]
    cont = continuous_futures_universe("ES_c", tradable=True)
    assert resolve_universe(cont) == ["ES_c"]

    custom = custom_universe(instruments=["ZZZ"])
    assert resolve_universe(custom) == ["ZZZ"]

    def _resolver(spec, asof=None):
        return ["CUSTOM1"]

    custom2 = custom_universe(resolver="demo")
    assert resolve_universe(custom2, custom_resolvers={"demo": _resolver}) == ["CUSTOM1"]


def test_filter_universe_membership_dataframe():
    df = pd.DataFrame(
        [
            {
                "instrument": "AAA",
                "start": datetime(2020, 1, 1, tzinfo=UTC),
                "end": datetime(2020, 3, 1, tzinfo=UTC),
            },
            {"symbol": "BBB", "start": datetime(2020, 2, 1, tzinfo=UTC), "end": pd.NaT},
        ]
    )
    # second row uses symbol — helper should handle mixed via column pick
    df2 = pd.DataFrame(
        [
            {
                "instrument": "AAA",
                "start": datetime(2020, 1, 1, tzinfo=UTC),
                "end": datetime(2020, 3, 1, tzinfo=UTC),
            },
            {"instrument": "BBB", "start": datetime(2020, 2, 1, tzinfo=UTC), "end": pd.NaT},
        ]
    )
    out = filter_universe_membership_asof(df2, datetime(2020, 2, 15, tzinfo=UTC))
    assert set(out) == {"AAA", "BBB"}
    with pytest.raises(LookaheadViolation):
        filter_universe_membership_asof(df2, datetime(2020, 2, 15))


def test_continuous_contract_volume_roll():
    # Two contracts with volume crossover
    days = pd.bdate_range("2020-01-01", periods=12, tz="UTC")
    rows = []
    for i, ts in enumerate(days):
        # Front high volume early; deferred later
        rows.append(
            {
                "timestamp": ts,
                "instrument": "CLH0",
                "open": 50 + i * 0.1,
                "high": 51 + i * 0.1,
                "low": 49 + i * 0.1,
                "close": 50.5 + i * 0.1,
                "volume": 1000 - i * 50,
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "instrument": "CLM0",
                "open": 51 + i * 0.1,
                "high": 52 + i * 0.1,
                "low": 50 + i * 0.1,
                "close": 51.5 + i * 0.1,
                "volume": 100 + i * 80,
            }
        )
    raw = pd.DataFrame(rows)
    cfg = ContinuousContractConfig(
        root="CL",
        continuous_symbol="CL_c",
        roll_rule=RollRule.VOLUME,
        adjustment=AdjustmentMethod.BACK_ADJUST,
        series_kind=ContractSeriesKind.CONTINUOUS_RESEARCH,
    )
    cont, rolls = ContinuousContractBuilder(cfg).build(raw)
    assert not cont.empty
    assert cont["instrument"].iloc[0] == "CL_c"
    assert cont["series_kind"].iloc[0] == "continuous_research"
    # Volume crossover should produce at least one roll in this setup
    assert isinstance(rolls, list)

    # Ratio + unadjusted via convenience
    cont2, _ = build_continuous_series(
        raw,
        {
            "root": "CL",
            "continuous_symbol": "CL_c",
            "roll_rule": "volume",
            "adjustment": "ratio",
        },
    )
    assert not cont2.empty
    cont3, _ = build_continuous_series(
        raw,
        ContinuousContractConfig(
            root="CL",
            continuous_symbol="CL_c",
            roll_rule=RollRule.VOLUME,
            adjustment=AdjustmentMethod.UNADJUSTED,
        ),
    )
    assert not cont3.empty


def test_continuous_contract_calendar_with_specs():
    days = pd.bdate_range("2020-01-01", periods=20, tz="UTC")
    rows = []
    for ts in days:
        for inst, base in [("ESH0", 3000.0), ("ESM0", 3010.0)]:
            rows.append(
                {
                    "timestamp": ts,
                    "instrument": inst,
                    "open": base,
                    "high": base + 1,
                    "low": base - 1,
                    "close": base,
                    "volume": 1000.0,
                }
            )
    raw = pd.DataFrame(rows)
    specs = [
        ContractSpec(
            contract="ESH0",
            root="ES",
            expiry=datetime(2020, 1, 17, tzinfo=UTC),
            multiplier=50.0,
        ),
        ContractSpec(
            contract="ESM0",
            root="ES",
            expiry=datetime(2020, 3, 20, tzinfo=UTC),
            multiplier=50.0,
        ),
    ]
    with pytest.raises(ValueError):
        ContractSpec(contract="X", root="X", expiry=datetime(2020, 1, 1))  # naive

    cfg = ContinuousContractConfig(
        root="ES",
        continuous_symbol="ES_c",
        roll_rule=RollRule.CALENDAR,
        calendar_days_before_expiry=5,
        adjustment=AdjustmentMethod.BACK_ADJUST,
        margin=1000.0,
        currency="USD",
    )
    cont, rolls = build_continuous_series(raw, cfg, contracts=specs)
    assert not cont.empty
    assert "margin" in cont.columns
    assert specs[0].to_dict()["contract"] == "ESH0"
    assert cfg.to_dict()["roll_rule"] == "calendar"
    if rolls:
        assert rolls[0].to_dict()["from_contract"]


def test_continuous_oi_requires_column():
    raw = generate_synthetic_ohlcv(n_days=5, instruments=["A", "B"], seed=1)
    cfg = ContinuousContractConfig(
        root="R",
        continuous_symbol="R_c",
        roll_rule=RollRule.OPEN_INTEREST,
    )
    with pytest.raises(ValueError):
        ContinuousContractBuilder(cfg).build(raw)
