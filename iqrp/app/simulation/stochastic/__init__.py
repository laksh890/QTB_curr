"""Stochastic path generators."""

from iqrp.app.simulation.base.generator import ensure_generators_loaded, get_generator_registry

ensure_generators_loaded()

__all__ = ["ensure_generators_loaded", "get_generator_registry"]
