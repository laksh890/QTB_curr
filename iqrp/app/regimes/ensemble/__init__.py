"""Institutional Ensemble Regime Intelligence Engine.

The sole regime interface for downstream forecasting, risk, portfolio, and execution.
"""

from __future__ import annotations

from iqrp.app.regimes.ensemble.calibration import Calibrator, brier_score, expected_calibration_error
from iqrp.app.regimes.ensemble.combiner import combine
from iqrp.app.regimes.ensemble.config import EnsembleSettings
from iqrp.app.regimes.ensemble.diagnostics import EnsembleDiagnostics
from iqrp.app.regimes.ensemble.evaluator import EnsembleEvaluator
from iqrp.app.regimes.ensemble.model import EnsembleRegimeModel, EnsembleStateSpaceModel
from iqrp.app.regimes.ensemble.registry import EnsembleMember, EnsembleRegistry, discover_modules
from iqrp.app.regimes.ensemble.serializer import EnsembleSerializer
from iqrp.app.regimes.ensemble.trainer import EnsembleTrainer
from iqrp.app.regimes.ensemble.weighting import compute_weights

__all__ = [
    "Calibrator",
    "EnsembleDiagnostics",
    "EnsembleEvaluator",
    "EnsembleMember",
    "EnsembleRegistry",
    "EnsembleRegimeModel",
    "EnsembleSerializer",
    "EnsembleSettings",
    "EnsembleStateSpaceModel",
    "EnsembleTrainer",
    "brier_score",
    "combine",
    "compute_weights",
    "discover_modules",
    "expected_calibration_error",
]
