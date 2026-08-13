"""Institutional Gaussian Mixture Regime Detection Engine.

Integrates with the State Space Framework and Probability Engine.
"""

from __future__ import annotations

from iqrp.app.regimes.gmm.config import GMMSettings
from iqrp.app.regimes.gmm.diagnostics import GMMDiagnostics
from iqrp.app.regimes.gmm.em import EMResult, fit_em
from iqrp.app.regimes.gmm.evaluator import GMMEvaluator
from iqrp.app.regimes.gmm.mixture import GaussianMixtureParams
from iqrp.app.regimes.gmm.model import GaussianMixtureModel, GMMRegimeModel
from iqrp.app.regimes.gmm.model_selection import select_n_components
from iqrp.app.regimes.gmm.serializer import GMMSerializer
from iqrp.app.regimes.gmm.trainer import GMMTrainer

__all__ = [
    "EMResult",
    "GMMDiagnostics",
    "GMMEvaluator",
    "GMMRegimeModel",
    "GMMSerializer",
    "GMMSettings",
    "GMMTrainer",
    "GaussianMixtureModel",
    "GaussianMixtureParams",
    "fit_em",
    "select_n_components",
]
