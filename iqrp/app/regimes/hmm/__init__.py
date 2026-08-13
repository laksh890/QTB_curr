"""Institutional Hidden Markov Model Engine.

Integrates with the State Space Framework and Probability Engine.
"""

from __future__ import annotations

from iqrp.app.regimes.hmm.baum_welch import BaumWelchResult, baum_welch
from iqrp.app.regimes.hmm.config import HMMSettings
from iqrp.app.regimes.hmm.diagnostics import HMMDiagnostics
from iqrp.app.regimes.hmm.emissions import (
    DiscreteEmissionModel,
    GaussianEmissionModel,
    build_emission,
)
from iqrp.app.regimes.hmm.evaluator import HMMEvaluator
from iqrp.app.regimes.hmm.forward import forward
from iqrp.app.regimes.hmm.forward_backward import forward_backward
from iqrp.app.regimes.hmm.model import HiddenMarkovModel, HMMRegimeModel
from iqrp.app.regimes.hmm.serializer import HMMSerializer
from iqrp.app.regimes.hmm.trainer import HMMTrainer
from iqrp.app.regimes.hmm.transitions import HMMTransitions
from iqrp.app.regimes.hmm.viterbi import viterbi

__all__ = [
    "BaumWelchResult",
    "DiscreteEmissionModel",
    "GaussianEmissionModel",
    "HMMDiagnostics",
    "HMMEvaluator",
    "HMMRegimeModel",
    "HMMSerializer",
    "HMMSettings",
    "HMMTrainer",
    "HMMTransitions",
    "HiddenMarkovModel",
    "baum_welch",
    "build_emission",
    "forward",
    "forward_backward",
    "viterbi",
]
