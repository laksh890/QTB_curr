"""Market Regime Detection Framework.

Downstream forecasting models must consume this package — never a concrete
algorithm (Markov, HMM, GMM, etc.) directly.
"""

from iqrp.app.regimes.base.evaluator import EvaluationReport, RegimeEvaluator
from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.persistence import PersistenceEngine, PersistenceReport
from iqrp.app.regimes.base.probabilities import ProbabilityBundle, ProbabilityEngine
from iqrp.app.regimes.base.regime import RegimeResult
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import (
    RegimeModelRegistry,
    ensure_regime_models_loaded,
    get_registry,
    regime_model_factory,
    register_regime_model,
)
from iqrp.app.regimes.base.state import RegimeState
from iqrp.app.regimes.base.transition import RegimeTransition
from iqrp.app.regimes.config import RegimeSettings
from iqrp.app.regimes.models.mock import MockRegimeModel
from iqrp.app.regimes.services.detector import RegimeDetector
from iqrp.app.regimes.services.predictor import RegimePredictor
from iqrp.app.regimes.services.serializer import RegimeSerializer
from iqrp.app.regimes.services.trainer import RegimeTrainer
from iqrp.app.regimes.storage.regime_store import RegimeStore
from iqrp.app.regimes.visualization import (
    plot_persistence,
    plot_probabilities,
    plot_timeline,
    plot_transitions,
)

ensure_regime_models_loaded()

__all__ = [
    "EvaluationReport",
    "MockRegimeModel",
    "PersistenceEngine",
    "PersistenceReport",
    "ProbabilityBundle",
    "ProbabilityEngine",
    "RegimeDetector",
    "RegimeEvaluator",
    "RegimeForecast",
    "RegimeModel",
    "RegimeModelMeta",
    "RegimeModelRegistry",
    "RegimePredictor",
    "RegimeResult",
    "RegimeSerializer",
    "RegimeSettings",
    "RegimeState",
    "RegimeStore",
    "RegimeTrainer",
    "RegimeTransition",
    "ensure_regime_models_loaded",
    "get_registry",
    "plot_persistence",
    "plot_probabilities",
    "plot_timeline",
    "plot_transitions",
    "regime_model_factory",
    "register_regime_model",
]
