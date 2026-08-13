"""Strategy registry and reference strategy tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from iqrp.app.backtesting.strategy import (
    BuyAndHoldStrategy,
    CrossSectionalMomentumStrategy,
    Strategy,
    StrategyRegistry,
    StrategyRegistryError,
)


class _V1(Strategy):
    strategy_id = "dual"
    strategy_version = "1.0.0"

    def initialize(self, context):
        return None


class _V2(Strategy):
    strategy_id = "dual"
    strategy_version = "2.0.0"

    def initialize(self, context):
        return None


class _NoId(Strategy):
    strategy_id = ""
    strategy_version = "1.0.0"

    def initialize(self, context):
        return None


def _event(ts: datetime, bars: dict | None = None):
    return SimpleNamespace(
        timestamp=ts,
        payload={"bars": bars or {"AAA": {"close": 10.0}, "BBB": {"close": 20.0}}},
    )


def test_registry_refuses_silent_selection():
    with pytest.raises(StrategyRegistryError, match="refusing silent"):
        StrategyRegistry.get("")
    with pytest.raises(StrategyRegistryError, match="unknown strategy"):
        StrategyRegistry.get("does_not_exist")


def test_registry_register_get_create():
    StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
    cls = StrategyRegistry.get("buy_and_hold")
    assert cls is BuyAndHoldStrategy
    inst = StrategyRegistry.create("buy_and_hold", "1.0.0", mode="equal_weight")
    assert isinstance(inst, BuyAndHoldStrategy)
    assert ("buy_and_hold", "1.0.0") in StrategyRegistry.registered()


def test_registry_duplicate_and_ambiguous():
    StrategyRegistry.register(_V1, overwrite=True)
    with pytest.raises(StrategyRegistryError, match="already registered"):
        StrategyRegistry.register(_V1, overwrite=False)
    StrategyRegistry.register(_V2, overwrite=True)
    with pytest.raises(StrategyRegistryError, match="multiple versions"):
        StrategyRegistry.get("dual")
    assert StrategyRegistry.get("dual", "2.0.0") is _V2
    with pytest.raises(StrategyRegistryError, match="unknown strategy"):
        StrategyRegistry.get("dual", "9.9.9")


def test_registry_rejects_empty_ids():
    with pytest.raises(ValueError):
        StrategyRegistry.register(_NoId)


def test_buy_and_hold_modes():
    ctx = SimpleNamespace(
        universe=["AAA", "BBB"],
        latest_prices={"AAA": 10.0, "BBB": 20.0},
        strategy_state={},
        positions=SimpleNamespace(open_instruments=lambda: []),
    )
    s = BuyAndHoldStrategy(mode="equal_weight")
    s.initialize(ctx)
    payload = s.on_market_data(_event(datetime(2020, 1, 2, tzinfo=UTC)), ctx)
    assert payload is not None
    assert payload.get("rebalance") is True
    payload2 = s.on_market_data(_event(datetime(2020, 1, 3, tzinfo=UTC)), ctx)
    assert payload2 is not None
    assert payload2.get("rebalance") is False
    with pytest.raises(ValueError):
        BuyAndHoldStrategy(mode="invalid")

    first = BuyAndHoldStrategy(mode="first_instrument")
    ctx2 = SimpleNamespace(
        universe=["AAA", "BBB"],
        latest_prices={"AAA": 10.0, "BBB": 20.0},
        strategy_state={},
        positions=SimpleNamespace(open_instruments=lambda: []),
    )
    first.initialize(ctx2)
    out = first.on_market_data(_event(datetime(2020, 1, 2, tzinfo=UTC)), ctx2)
    assert out is not None
    assert out["target_weights"].get("AAA") == 1.0
    feats = first.on_features(_event(datetime(2020, 1, 2, tzinfo=UTC)), ctx2)
    assert feats is not None
    end = first.on_end(ctx2)
    assert end["entered"] is True


def test_cross_sectional_momentum():
    s = CrossSectionalMomentumStrategy(lookback=5, top_n=1, long_only=True)
    ctx = SimpleNamespace(latest_prices={}, strategy_state={})
    s.initialize(ctx)
    out = None
    for i in range(12):
        prices = {"AAA": 100.0 + i * 2, "BBB": 100.0 - i * 0.5}
        ctx.latest_prices = prices
        bars = {k: {"close": v} for k, v in prices.items()}
        out = s.on_features(_event(datetime(2020, 1, 2 + i, tzinfo=UTC), bars), ctx)
    assert out is not None
    assert "target_weights" in out
    sig = s.on_signal(_event(datetime(2020, 1, 20, tzinfo=UTC)), ctx)
    assert sig is not None
    assert s.on_end(ctx)["n_history"] >= 1

    shortable = CrossSectionalMomentumStrategy(lookback=3, long_only=False)
    ctx2 = SimpleNamespace(latest_prices={}, strategy_state={})
    shortable.initialize(ctx2)
    for i in range(8):
        prices = {"A": 50.0 + i, "B": 50.0 - i, "C": 50.0}
        ctx2.latest_prices = prices
        bars = {k: {"close": v} for k, v in prices.items()}
        shortable.on_features(_event(datetime(2020, 1, 2 + i, tzinfo=UTC), bars), ctx2)
