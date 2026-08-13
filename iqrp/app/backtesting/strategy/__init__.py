"""Strategy interfaces and reference implementations for operational backtests."""

from iqrp.app.backtesting.strategy.base import Strategy
from iqrp.app.backtesting.strategy.buy_and_hold import BuyAndHoldStrategy
from iqrp.app.backtesting.strategy.cross_sectional_momentum import CrossSectionalMomentumStrategy
from iqrp.app.backtesting.strategy.registry import StrategyRegistry, StrategyRegistryError

__all__ = [
    "Strategy",
    "StrategyRegistry",
    "StrategyRegistryError",
    "BuyAndHoldStrategy",
    "CrossSectionalMomentumStrategy",
]
