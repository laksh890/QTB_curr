"""Shared fixtures for Institutional Alpha Research unit tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_registry import SignalRegistry
from iqrp.app.alpha.config import AlphaSettings
from iqrp.app.alpha.engine import AlphaResearchEngine
from iqrp.app.alpha.processes import (
    available_scenarios,
    decaying_signal,
    genuine_momentum,
    random_noise,
    regime_specific,
    simulate_alpha_scenario,
)
from iqrp.app.alpha.research.decay import forward_returns
from iqrp.app.alpha.statistical_validation.multiple_testing import get_experiment_tracker

SEED = 42
N = 300
N_TRIALS = 20
N_BOOT = 60
N_PERM = 60


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def n() -> int:
    return N


@pytest.fixture
def n_trials() -> int:
    return N_TRIALS


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def alpha_settings() -> AlphaSettings:
    return AlphaSettings(
        seed=SEED,
        scoring={"allow_sharpe_only_approval": False, "min_hypothesis_chars": 20},
        research={"horizons": (1, 2, 5, 10), "stability_window": 60},
        discovery={"auto_register": True},
        governance={"preserve_rejected": True},
    )


@pytest.fixture
def registry() -> SignalRegistry:
    """Fresh registry per test — avoid default singleton leakage."""
    return SignalRegistry()


@pytest.fixture
def engine(alpha_settings: AlphaSettings, registry: SignalRegistry) -> AlphaResearchEngine:
    return AlphaResearchEngine(settings=alpha_settings, registry=registry)


@pytest.fixture(autouse=True)
def _reset_mt_tracker() -> None:
    from iqrp.app.alpha.base.signal_registry import get_default_registry

    get_experiment_tracker().reset()
    get_default_registry().clear()
    yield
    get_experiment_tracker().reset()
    get_default_registry().clear()


@pytest.fixture
def genuine(n: int, seed: int) -> dict[str, Any]:
    return genuine_momentum(n, seed=seed, lookback=20, signal_strength=0.35)


@pytest.fixture
def noise(n: int, seed: int) -> dict[str, Any]:
    return random_noise(n, seed=seed)


@pytest.fixture
def regime_scen(n: int, seed: int) -> dict[str, Any]:
    return regime_specific(n, seed=seed)


@pytest.fixture
def decay_scen(n: int, seed: int) -> dict[str, Any]:
    return decaying_signal(n, seed=seed, half_life=5.0)


@pytest.fixture
def scenario_genuine(n: int, seed: int) -> dict[str, Any]:
    return simulate_alpha_scenario("genuine_momentum", n=n, seed=seed)


@pytest.fixture
def scenario_noise(n: int, seed: int) -> dict[str, Any]:
    return simulate_alpha_scenario("random_noise", n=n, seed=seed)


@pytest.fixture
def returns(genuine: dict[str, Any]) -> np.ndarray:
    return np.asarray(genuine["returns"], dtype=np.float64)


@pytest.fixture
def signal(genuine: dict[str, Any]) -> np.ndarray:
    return np.asarray(genuine["signal"], dtype=np.float64)


@pytest.fixture
def fwd(returns: np.ndarray) -> np.ndarray:
    return forward_returns(returns, 1)


@pytest.fixture
def alpha_signal(signal: np.ndarray) -> AlphaSignal:
    return AlphaSignal(values=signal, name="fixture_momentum")


@pytest.fixture
def hypothesis() -> str:
    return (
        "Short-horizon continuation arises from underreaction to information "
        "and gradual capital flows; this is a research hypothesis, not proof."
    )


@pytest.fixture
def definition(hypothesis: str) -> SignalDefinition:
    return SignalDefinition(
        name="test_momentum",
        version="1.0.0",
        formula="sum(returns, 20)",
        features=("returns",),
        lookback=20,
        horizon=1,
        universe="default",
        frequency="1d",
        direction="long_short",
        expected_relationship="positive",
        economic_hypothesis=hypothesis,
        owner="research",
        signal_type="momentum",
        tags=("unit_test",),
    )


@pytest.fixture
def thin_definition() -> SignalDefinition:
    return SignalDefinition(
        name="thin_hyp",
        version="1.0.0",
        formula="x",
        features=("x",),
        lookback=5,
        horizon=1,
        universe="default",
        frequency="1d",
        direction="long_short",
        expected_relationship="unknown",
        economic_hypothesis="",
        owner="research",
    )


@pytest.fixture
def panel(rng: np.random.Generator) -> np.ndarray:
    """T x N cross-section panel."""
    return rng.normal(0.0, 1.0, size=(80, 20))


@pytest.fixture
def sectors() -> np.ndarray:
    return np.array(["A"] * 10 + ["B"] * 10, dtype=object)


@pytest.fixture
def all_scenario_names() -> list[str]:
    return available_scenarios()
