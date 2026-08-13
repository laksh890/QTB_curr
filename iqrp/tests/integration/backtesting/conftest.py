"""Integration fixtures for operational backtesting E2E flows."""

from __future__ import annotations

import pytest

from iqrp.app.backtesting.strategy import (
    BuyAndHoldStrategy,
    CrossSectionalMomentumStrategy,
    StrategyRegistry,
)


@pytest.fixture(autouse=True)
def _clean_strategy_registry():
    StrategyRegistry.clear()
    StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
    StrategyRegistry.register(CrossSectionalMomentumStrategy, overwrite=True)
    yield
    StrategyRegistry.clear()
