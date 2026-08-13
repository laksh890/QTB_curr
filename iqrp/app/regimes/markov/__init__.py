"""Institutional Markov Chain Engine.

First concrete State Space Framework implementation for discrete-time,
time-homogeneous Markov chains. Also exposes a :class:`RegimeModel` adapter.
"""

from __future__ import annotations

from iqrp.app.regimes.markov.config import MarkovSettings
from iqrp.app.regimes.markov.diagnostics import MarkovDiagnostics
from iqrp.app.regimes.markov.estimator import TransitionEstimator
from iqrp.app.regimes.markov.evaluator import MarkovEvaluator
from iqrp.app.regimes.markov.forecast import MarkovForecaster
from iqrp.app.regimes.markov.model import MarkovChainModel, MarkovRegimeModel
from iqrp.app.regimes.markov.persistence import PersistenceAnalyzer
from iqrp.app.regimes.markov.serializer import MarkovSerializer
from iqrp.app.regimes.markov.state_mapper import LabelStateMapper
from iqrp.app.regimes.markov.stationary import StationaryAnalyzer
from iqrp.app.regimes.markov.trainer import MarkovTrainer
from iqrp.app.regimes.markov.transition import TransitionMatrix

__all__ = [
    "LabelStateMapper",
    "MarkovChainModel",
    "MarkovDiagnostics",
    "MarkovEvaluator",
    "MarkovForecaster",
    "MarkovRegimeModel",
    "MarkovSerializer",
    "MarkovSettings",
    "MarkovTrainer",
    "PersistenceAnalyzer",
    "StationaryAnalyzer",
    "TransitionEstimator",
    "TransitionMatrix",
]
