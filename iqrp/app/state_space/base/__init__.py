"""State-space base contracts and value objects."""

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.latent_state import LatentState
from iqrp.app.state_space.base.observation import Observation
from iqrp.app.state_space.base.observation_model import (
    DiagonalGaussianObservationModel,
    ObservationModel,
)
from iqrp.app.state_space.base.registry import (
    ensure_state_space_models_loaded,
    get_registry,
    register_state_space_model,
)
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta
from iqrp.app.state_space.base.transition_model import MatrixTransitionModel, TransitionModel

__all__ = [
    "DiagonalGaussianObservationModel",
    "FilterResult",
    "ForecastResult",
    "LatentState",
    "MatrixTransitionModel",
    "Observation",
    "ObservationModel",
    "SmootherResult",
    "StateSpaceModel",
    "StateSpaceModelMeta",
    "TransitionModel",
    "ensure_state_space_models_loaded",
    "get_registry",
    "register_state_space_model",
]
