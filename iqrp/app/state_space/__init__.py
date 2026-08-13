"""Institutional State Space Modeling Framework.

Downstream algorithms (Markov, HMM, Kalman, particle, DLM, …) plug into
:class:`~iqrp.app.state_space.base.state_space_model.StateSpaceModel`.
"""

from iqrp.app.state_space.base import (
    DiagonalGaussianObservationModel,
    FilterResult,
    ForecastResult,
    LatentState,
    MatrixTransitionModel,
    Observation,
    ObservationModel,
    SmootherResult,
    StateSpaceModel,
    StateSpaceModelMeta,
    TransitionModel,
    ensure_state_space_models_loaded,
    get_registry,
    register_state_space_model,
)
from iqrp.app.state_space.config import StateSpaceSettings
from iqrp.app.state_space.evaluation import EvaluationMetrics, StateSpaceDiagnostics
from iqrp.app.state_space.filtering import BackwardFilter, ForwardFilter
from iqrp.app.state_space.forecasting import MultiStepForecaster
from iqrp.app.state_space.smoothing import FixedIntervalSmoother, FixedLagSmoother
from iqrp.app.state_space.storage import StateSpaceSerializer, StateStore

ensure_state_space_models_loaded()

__all__ = [
    "BackwardFilter",
    "DiagonalGaussianObservationModel",
    "EvaluationMetrics",
    "FilterResult",
    "FixedIntervalSmoother",
    "FixedLagSmoother",
    "ForecastResult",
    "ForwardFilter",
    "LatentState",
    "MatrixTransitionModel",
    "MultiStepForecaster",
    "Observation",
    "ObservationModel",
    "SmootherResult",
    "StateSpaceDiagnostics",
    "StateSpaceModel",
    "StateSpaceModelMeta",
    "StateSpaceSerializer",
    "StateSpaceSettings",
    "StateStore",
    "TransitionModel",
    "ensure_state_space_models_loaded",
    "get_registry",
    "register_state_space_model",
]
