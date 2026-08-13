"""Institutional Market Simulation Engine.

Purpose: validation, benchmarking, stress testing, and algorithm development.
Downstream modules must consume synthetic markets from this package rather than
hand-rolling ad-hoc random walks.
"""

from iqrp.app.simulation.base.generator import (
    GeneratorMeta,
    PathGenerator,
    PathResult,
    ensure_generators_loaded,
    get_generator_registry,
    register_generator,
)
from iqrp.app.simulation.base.market import GroundTruth, SimulatedMarket
from iqrp.app.simulation.base.scenario import Scenario
from iqrp.app.simulation.base.simulator import MarketSimulator
from iqrp.app.simulation.config import SimulationSettings
from iqrp.app.simulation.noise import NoiseSampler, sample_noise
from iqrp.app.simulation.portfolio import ExecutionReport, SimulatedExecutionEngine
from iqrp.app.simulation.regimes import (
    HiddenRegimeSimulator,
    RegimeSwitchingSimulator,
)
from iqrp.app.simulation.validation import SimulationValidator, ValidationReport
from iqrp.app.simulation.visualization import write_all_charts

ensure_generators_loaded()

__all__ = [
    "ExecutionReport",
    "GeneratorMeta",
    "GroundTruth",
    "HiddenRegimeSimulator",
    "MarketSimulator",
    "NoiseSampler",
    "PathGenerator",
    "PathResult",
    "RegimeSwitchingSimulator",
    "Scenario",
    "SimulatedExecutionEngine",
    "SimulatedMarket",
    "SimulationSettings",
    "SimulationValidator",
    "ValidationReport",
    "ensure_generators_loaded",
    "get_generator_registry",
    "register_generator",
    "sample_noise",
    "write_all_charts",
]
