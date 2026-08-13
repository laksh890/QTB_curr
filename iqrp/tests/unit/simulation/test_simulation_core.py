"""Unit tests for simulation primitives and generators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.simulation import (
    GroundTruth,
    MarketSimulator,
    NoiseSampler,
    Scenario,
    SimulatedExecutionEngine,
    SimulationSettings,
    SimulationValidator,
    ensure_generators_loaded,
    get_generator_registry,
    sample_noise,
    write_all_charts,
)
from iqrp.app.simulation.base.generator import GeneratorRegistry
from iqrp.app.simulation.events import apply_event_suite
from iqrp.app.simulation.liquidity import OrderBookGenerator, SlippageModel, SpreadModel
from iqrp.app.simulation.regimes import HiddenRegimeSimulator, RegimeSwitchingSimulator


@pytest.mark.unit
def test_settings_and_registry() -> None:
    settings = SimulationSettings.from_hydra(overrides=["n_steps=250", "default_model=ou"])
    assert settings.n_steps == 250
    assert settings.default_model == "ou"
    assert SimulationSettings.default().enabled is True
    ensure_generators_loaded()
    names = get_generator_registry().list_names()
    for required in (
        "gbm",
        "abm",
        "ou",
        "merton_jump",
        "jump_diffusion",
        "heston",
        "variance_gamma",
        "cir",
        "random_walk",
    ):
        assert required in names
    with pytest.raises(ConfigurationError):
        get_generator_registry().get_class("nope")
    with pytest.raises(ConfigurationError):
        SimulationSettings.from_mapping("bad")  # type: ignore[arg-type]


@pytest.mark.unit
def test_all_generators_produce_finite_paths() -> None:
    rng = np.random.default_rng(0)
    for name in get_generator_registry().list_names():
        gen = get_generator_registry().create(name, rng=rng)
        result = gen.generate(120, x0=100.0, dt=0.01, volatility=0.2, drift=0.05)
        assert np.all(np.isfinite(result.prices))
        assert len(result.returns) == 120
        assert np.all(result.prices > 0)


@pytest.mark.unit
def test_noise_distributions() -> None:
    rng = np.random.default_rng(1)
    for dist in ("gaussian", "student_t", "laplace", "cauchy", "uniform", "mixture"):
        z = sample_noise(500, dist, rng=rng)  # type: ignore[arg-type]
        assert z.shape == (500,)
        assert np.all(np.isfinite(z))
    sampler = NoiseSampler("mixture", rng=rng)
    assert sampler.sample((10, 2)).shape == (10, 2)
    with pytest.raises(ValidationError):
        sample_noise(3, "nope", rng=rng)  # type: ignore[arg-type]


@pytest.mark.unit
def test_regime_and_hidden() -> None:
    rng = np.random.default_rng(2)
    tm = RegimeSwitchingSimulator.mixed_transition(3, 0.9)
    path = RegimeSwitchingSimulator(rng).simulate(
        200,
        transition_matrix=tm,
        state_names=("bear", "sideways", "bull"),
        drifts=(-0.1, 0.0, 0.1),
        volatilities=(0.3, 0.1, 0.2),
    )
    assert path.state_ids.shape == (200,)
    assert path.transition_matrix.shape == (3, 3)
    hidden = HiddenRegimeSimulator(rng).simulate(
        100,
        transition_matrix=tm,
        state_names=("a", "b", "c"),
        emission_means=(-0.01, 0.0, 0.01),
        emission_stds=(0.02, 0.01, 0.02),
    )
    assert hidden.observations.shape == (100,)


@pytest.mark.unit
def test_microstructure_and_events() -> None:
    rng = np.random.default_rng(3)
    mids = np.linspace(100, 110, 80)
    vol = np.full(80, 0.2)
    spreads = SpreadModel(rng=rng).spreads_bps(mids, vol)
    bid, ask = SpreadModel(rng=rng).bid_ask(mids, spreads)
    assert np.all(ask >= bid)
    book = OrderBookGenerator(rng=rng).generate_frame(
        mids, spreads * mids / 10_000, list(range(80)), stride=5
    )
    assert book.height > 0
    trades = OrderBookGenerator(rng=rng).generate_trades(mids, np.ones(80) * 10, list(range(80)))
    assert trades.height == 80 * 3
    fill = SlippageModel(rng=rng).execution_price(
        100.0, "buy", 1.0, adv=1000.0, volatility=0.2, spread=0.05
    )
    assert fill["price"] > 100.0
    p, _v, _vol2, _s, masks = apply_event_suite(
        mids,
        np.ones(80) * 50,
        vol,
        spreads,
        rng=rng,
        flash_crash_prob=0.2,
        news_shock_prob=0.2,
        gap_open_prob=0.2,
        liquidity_collapse_prob=0.2,
        outage_prob=0.1,
        vol_spike_prob=0.2,
        momentum_burst_prob=0.2,
    )
    assert len(p) == 80
    assert "flash_crash" in masks


@pytest.mark.unit
def test_simulator_end_to_end(tmp_path: Path) -> None:
    settings = SimulationSettings.model_validate(
        {
            **SimulationSettings.default().model_dump(),
            "n_steps": 180,
            "visualization": {
                "enabled": True,
                "max_points": 200,
                "output_dir": str(tmp_path / "charts"),
            },
            "events": {
                **SimulationSettings.default().events.model_dump(),
                "enabled": True,
                "flash_crash_prob": 0.01,
            },
        }
    )
    sim = MarketSimulator(settings)
    assert "gbm" in sim.available_models()
    market = sim.simulate(
        Scenario.from_settings(settings, name="unit", model="gbm"),
        write_charts=True,
        validate=True,
    )
    assert market.ohlcv().height == 181
    assert market.ground_truth.regime_ids.shape[0] == 180
    assert market.trades.height > 0
    assert "validation" in market.metadata
    assert (tmp_path / "charts" / "unit" / "price.svg").exists()
    assert market.to_dict()["model"] == "gbm"
    assert market.ground_truth.to_frame().height == 180


@pytest.mark.unit
def test_multi_asset_and_presets(tmp_path: Path) -> None:
    settings = SimulationSettings.model_validate(
        {
            **SimulationSettings.default().model_dump(),
            "n_steps": 100,
            "n_assets": 3,
            "visualization": {"enabled": False, "max_points": 100, "output_dir": str(tmp_path)},
        }
    )
    sim = MarketSimulator(settings)
    market = sim.simulate(Scenario.from_settings(settings, model="gbm"), validate=False)
    assert market.candles["symbol"].n_unique() == 3
    for preset in ("bull", "bear", "sideways", "high_volatility", "mixed"):
        m = sim.simulate_preset(preset, n_steps=60, model="gbm")
        assert m.candles.height == 61


@pytest.mark.unit
def test_execution_and_validator() -> None:
    sim = MarketSimulator(
        SimulationSettings.model_validate(
            {
                **SimulationSettings.default().model_dump(),
                "n_steps": 120,
                "events": {"enabled": False},
            }
        )
    )
    market = sim.simulate(validate=True)
    eng = SimulatedExecutionEngine(impact=0.1, rng=np.random.default_rng(0))
    mids = market.ohlcv()["close"].to_numpy()
    spreads = market.ohlcv()["spread_bps"].to_numpy() * mids / 10_000
    report = eng.execute_twap(
        symbol=market.symbols[0],
        side="buy",
        quantity=5.0,
        mids=mids,
        spreads=spreads,
        volumes=market.ohlcv()["volume"].to_numpy(),
        volatility=market.ground_truth.volatility,
        timestamps=market.timestamps,
        n_slices=8,
    )
    assert report.frame.height == 8
    assert report.total_notional > 0
    assert "n_fills" in report.to_dict()
    val = SimulationValidator().validate_returns(
        market.returns(),
        expected_drift=0.05,
        expected_volatility=0.2,
        dt=0.004,
    )
    assert "n" in val.to_dict()["details"]
    acf = SimulationValidator().autocorrelation(market.returns(), lags=5)
    assert len(acf) == 6


@pytest.mark.unit
def test_disabled_and_registry_meta() -> None:
    disabled = SimulationSettings.model_validate(
        {**SimulationSettings.default().model_dump(), "enabled": False}
    )
    with pytest.raises(ConfigurationError):
        MarketSimulator(disabled)
    local = GeneratorRegistry()

    class Bad:
        pass

    with pytest.raises(ConfigurationError):
        local.register(Bad)  # type: ignore[arg-type]
    meta = get_generator_registry().describe("gbm")
    assert meta.to_dict()["name"] == "gbm"


@pytest.mark.unit
def test_ground_truth_multi_dim() -> None:
    gt = GroundTruth(
        regime_ids=np.array([0, 1, 1]),
        regime_names=("a", "b"),
        volatility=np.array([[0.1], [0.2], [0.2]]),
        drift=np.array([[0.0], [0.1], [0.1]]),
        trend=np.array([[0.0], [1.0], [1.0]]),
        transition_matrix=np.eye(2),
        event_mask={"news": np.array([0, 1, 0])},
    )
    assert gt.to_frame().height == 3
    assert "volatility" in gt.to_dict()


@pytest.mark.unit
@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=30, max_value=80), st.floats(0.05, 0.4))
def test_property_gbm_positive_prices(n: int, vol: float) -> None:
    gen = get_generator_registry().create("gbm", rng=np.random.default_rng(n))
    result = gen.generate(n, x0=50.0, dt=0.01, volatility=vol, drift=0.0)
    assert np.all(result.prices > 0)
    assert result.returns.shape[0] == n


@pytest.mark.unit
def test_heston_vg_cir_jump(tmp_path: Path) -> None:
    sim = MarketSimulator(
        SimulationSettings.model_validate(
            {
                **SimulationSettings.default().model_dump(),
                "n_steps": 90,
                "regimes": {**SimulationSettings.default().regimes.model_dump(), "enabled": False},
                "events": {**SimulationSettings.default().events.model_dump(), "enabled": False},
                "visualization": {
                    "enabled": True,
                    "max_points": 90,
                    "output_dir": str(tmp_path),
                },
            }
        )
    )
    for model in ("heston", "variance_gamma", "cir", "merton_jump", "ou", "abm", "random_walk"):
        market = sim.simulate(
            Scenario.from_settings(sim.settings, name=model, model=model),
            write_charts=False,
            validate=False,
        )
        assert market.returns().shape[0] == 90
    market = sim.simulate(
        Scenario.from_settings(sim.settings, name="viz", model="gbm"),
        write_charts=True,
        validate=False,
    )
    paths = write_all_charts(market, tmp_path / "all")
    assert len(paths) == 7
