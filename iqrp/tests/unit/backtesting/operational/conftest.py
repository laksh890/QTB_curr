"""Fixtures for Phase 13 operational backtesting tests.

Synthetic OHLCV is generated via ``write_synthetic_ohlcv`` under
``iqrp/tests/fixtures/backtesting/`` (not committed as large binaries).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv
from iqrp.app.backtesting.strategy import (
    BuyAndHoldStrategy,
    CrossSectionalMomentumStrategy,
    StrategyRegistry,
)

FIXTURES_ROOT = Path(__file__).resolve().parents[3] / "fixtures" / "backtesting"
SEED = 7


@pytest.fixture(autouse=True)
def _clean_strategy_registry():
    StrategyRegistry.clear()
    yield
    StrategyRegistry.clear()


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """Per-test writable fixture dir (avoids cross-test pollution)."""
    d = tmp_path / "fixtures"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def shared_fixtures_dir() -> Path:
    FIXTURES_ROOT.mkdir(parents=True, exist_ok=True)
    return FIXTURES_ROOT


@pytest.fixture
def synthetic_parquet(fixtures_dir: Path, seed: int) -> Path:
    path = fixtures_dir / "synthetic_bars.parquet"
    write_synthetic_ohlcv(
        path,
        n_days=40,
        instruments=["AAA", "BBB"],
        seed=seed,
        start="2020-01-01",
    )
    return path


@pytest.fixture
def synthetic_csv(fixtures_dir: Path, seed: int) -> Path:
    path = fixtures_dir / "synthetic_bars.csv"
    write_synthetic_ohlcv(
        path,
        n_days=30,
        instruments=["AAA", "BBB"],
        seed=seed,
        start="2020-01-01",
    )
    return path


@pytest.fixture
def synthetic_feather(fixtures_dir: Path, seed: int) -> Path:
    path = fixtures_dir / "synthetic_bars.feather"
    write_synthetic_ohlcv(
        path,
        n_days=25,
        instruments=["AAA"],
        seed=seed,
        start="2020-01-01",
    )
    return path


@pytest.fixture
def registered_strategies():
    StrategyRegistry.register(BuyAndHoldStrategy, overwrite=True)
    StrategyRegistry.register(CrossSectionalMomentumStrategy, overwrite=True)
    return StrategyRegistry


@pytest.fixture
def buy_and_hold_config(synthetic_parquet: Path, tmp_path: Path, seed: int) -> dict:
    return {
        "backtest_id": "op_unit",
        "strategy_id": "buy_and_hold",
        "strategy_version": "1.0.0",
        "strategy_params": {"mode": "equal_weight"},
        "dataset_path": str(synthetic_parquet),
        "dataset_id": "op_unit",
        "adapter": "parquet",
        "start": "2020-01-01",
        "end": "2020-02-28",
        "initial_capital": 1_000_000.0,
        "seed": seed,
        "output_dir": str(tmp_path / "results"),
        "spread_bps": 1.0,
        "enforce_pit": True,
        "risk_config": {"max_gross_leverage": 1.0},
    }
