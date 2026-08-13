"""Institutional Particle Filter Engine (Sequential Monte Carlo).

Integrates with the State Space Framework and Probability Engine.
"""

from __future__ import annotations

from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.diagnostics import ParticleDiagnostics
from iqrp.app.regimes.particle.evaluator import ParticleEvaluator
from iqrp.app.regimes.particle.model import ParticleFilterModel, ParticleRegimeModel
from iqrp.app.regimes.particle.particle import FilterTrace, Particle, ParticleCloud
from iqrp.app.regimes.particle.propagation import TransitionModel, build_transition
from iqrp.app.regimes.particle.resampling import (
    adaptive_resample,
    apply_resampling,
    multinomial_resample,
    residual_resample,
    stratified_resample,
)
from iqrp.app.regimes.particle.serializer import ParticleSerializer
from iqrp.app.regimes.particle.smoothing import SmoothTrace, trajectory_smooth
from iqrp.app.regimes.particle.trainer import (
    ParticleTrainer,
    filter_adaptive,
    filter_auxiliary,
    filter_bootstrap,
    filter_rao_blackwellized,
    filter_sir,
    filter_sis,
    run_filter,
    simulate_nonlinear,
)
from iqrp.app.regimes.particle.weighting import effective_sample_size, log_likelihood

__all__ = [
    "FilterTrace",
    "Particle",
    "ParticleCloud",
    "ParticleDiagnostics",
    "ParticleEvaluator",
    "ParticleFilterModel",
    "ParticleRegimeModel",
    "ParticleSerializer",
    "ParticleSettings",
    "ParticleTrainer",
    "SmoothTrace",
    "TransitionModel",
    "adaptive_resample",
    "apply_resampling",
    "build_transition",
    "effective_sample_size",
    "filter_adaptive",
    "filter_auxiliary",
    "filter_bootstrap",
    "filter_rao_blackwellized",
    "filter_sir",
    "filter_sis",
    "log_likelihood",
    "multinomial_resample",
    "residual_resample",
    "run_filter",
    "simulate_nonlinear",
    "stratified_resample",
    "trajectory_smooth",
]
