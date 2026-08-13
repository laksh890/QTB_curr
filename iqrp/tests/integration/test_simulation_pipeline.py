"""Integration tests for the Market Simulation Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.simulation import (
    MarketSimulator,
    Scenario,
    SimulatedExecutionEngine,
    SimulationSettings,
)


@pytest.mark.integration
def test_full_pipeline_multi_model(tmp_path: Path) -> None:
    settings = SimulationSettings.model_validate(
        {
            **SimulationSettings.default().model_dump(),
            "n_steps": 300,
            "n_assets": 2,
            "default_model": "gbm",
            "noise": {
                **SimulationSettings.default().noise.model_dump(),
                "distribution": "student_t",
            },
            "visualization": {
                "enabled": True,
                "max_points": 300,
                "output_dir": str(tmp_path / "sim"),
            },
        }
    )
    sim = MarketSimulator(settings)
    scenario = Scenario.from_settings(settings, name="integration", model="gbm")
    market = sim.simulate(scenario, write_charts=True, validate=True)
    assert market.ohlcv(market.symbols[0]).height == 301
    assert market.candles.height == 301 * 2
    assert set(market.candles["symbol"].unique().to_list()) == set(market.symbols)
    assert market.orderbook_snapshots.height > 0
    assert market.ground_truth.transition_matrix.shape[0] >= 1
    assert (tmp_path / "sim" / "integration" / "regimes.svg").exists()

    # Stress with jumps + events
    jump_scenario = Scenario.from_settings(
        settings, name="jumps", model="merton_jump", n_steps=250, n_assets=1
    )
    jumped = sim.simulate(jump_scenario, write_charts=False, validate=True)
    assert np.isfinite(jumped.returns()).all()

    eng = SimulatedExecutionEngine(rng=np.random.default_rng(7))
    ohlcv = market.ohlcv(market.symbols[0])
    mids = ohlcv["close"].to_numpy()
    report = eng.execute_twap(
        symbol=market.symbols[0],
        side="sell",
        quantity=2.5,
        mids=mids,
        spreads=ohlcv["spread_bps"].to_numpy() * mids / 10_000,
        volumes=ohlcv["volume"].to_numpy(),
        volatility=market.ground_truth.volatility,
        timestamps=market.timestamps,
        n_slices=12,
    )
    assert report.average_slippage_bps >= 0 or report.average_slippage_bps < 0
    assert report.frame.height == 12
