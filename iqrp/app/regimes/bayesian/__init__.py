"""Institutional Bayesian Regime Switching Engine.

Integrates with the State Space Framework and Probability Engine.
"""

from __future__ import annotations

from iqrp.app.regimes.bayesian.config import BayesianSettings
from iqrp.app.regimes.bayesian.diagnostics import BayesianDiagnostics
from iqrp.app.regimes.bayesian.evaluator import BayesianEvaluator
from iqrp.app.regimes.bayesian.gibbs import run_gibbs
from iqrp.app.regimes.bayesian.hmc import run_hmc
from iqrp.app.regimes.bayesian.metropolis import run_metropolis
from iqrp.app.regimes.bayesian.model import BayesianRegimeModel, BayesianRegimeSwitchingModel
from iqrp.app.regimes.bayesian.posterior import Posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.serializer import BayesianSerializer
from iqrp.app.regimes.bayesian.trainer import BayesianTrainer
from iqrp.app.regimes.bayesian.variational import run_variational

__all__ = [
    "BayesianDiagnostics",
    "BayesianEvaluator",
    "BayesianRegimeModel",
    "BayesianRegimeSwitchingModel",
    "BayesianSerializer",
    "BayesianSettings",
    "BayesianTrainer",
    "ModelPriors",
    "Posterior",
    "run_gibbs",
    "run_hmc",
    "run_metropolis",
    "run_variational",
]
