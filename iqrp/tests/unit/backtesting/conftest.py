"""Shared fixtures for Institutional Backtesting Platform unit tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from iqrp.app.backtesting.config import BacktestSettings, CostsConfig
from iqrp.app.backtesting.engine import BacktestEngine

SEED = 42


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def returns(rng: np.random.Generator) -> np.ndarray:
    """Synthetic daily returns (~252 bars)."""
    return rng.normal(0.0004, 0.01, size=252)


@pytest.fixture
def short_returns(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.001, 0.015, size=60)


@pytest.fixture
def prices(returns: np.ndarray) -> np.ndarray:
    return 100.0 * np.cumprod(1.0 + returns)


@pytest.fixture
def multi_prices(rng: np.random.Generator) -> np.ndarray:
    """(T+1, N) price matrix for 3 assets."""
    rets = rng.normal(0.0003, 0.012, size=(120, 3))
    return 100.0 * np.cumprod(1.0 + rets, axis=0)


@pytest.fixture
def signals(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=252)


@pytest.fixture
def timestamps() -> list[datetime]:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return [start + timedelta(days=i) for i in range(60)]


@pytest.fixture
def index_timestamps() -> list[int]:
    return list(range(100))


@pytest.fixture
def settings() -> BacktestSettings:
    return BacktestSettings(
        name="unit_test",
        initial_cash=1_000_000.0,
        costs=CostsConfig(commission_bps=1.0, spread_bps=1.0, slippage_bps=1.0),
    )


@pytest.fixture
def engine(settings: BacktestSettings) -> BacktestEngine:
    return BacktestEngine(settings=settings)


@pytest.fixture
def membership() -> dict[str, tuple[int, int | None]]:
    return {
        "AAA": (0, 80),
        "BBB": (10, None),
        "CCC": (50, 90),
        "DDD": (100, None),  # listed after asof=50
    }


@pytest.fixture
def trade_list() -> list[dict[str, Any]]:
    return [
        {"pnl": 10.0, "holding": 5, "side": "long"},
        {"pnl": -4.0, "holding": 2, "side": "short"},
        {"pnl": 7.0, "holding": 3, "side": "long"},
        {"pnl": -2.0, "holding": 1, "side": "long"},
        {"pnl": 3.0, "holding": 4, "side": "short"},
    ]
