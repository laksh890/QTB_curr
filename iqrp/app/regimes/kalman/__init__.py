"""Institutional Kalman Filtering Engine.

Integrates with the State Space Framework and Probability Engine.
"""

from __future__ import annotations

from iqrp.app.regimes.kalman.adaptive import adapt_noise_from_trace, filter_adaptive
from iqrp.app.regimes.kalman.config import KalmanSettings
from iqrp.app.regimes.kalman.diagnostics import KalmanDiagnostics
from iqrp.app.regimes.kalman.ekf import filter_ekf
from iqrp.app.regimes.kalman.evaluator import KalmanEvaluator
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM, build_system
from iqrp.app.regimes.kalman.linear import FilterTrace, filter_linear
from iqrp.app.regimes.kalman.model import KalmanFilterModel, KalmanRegimeModel
from iqrp.app.regimes.kalman.serializer import KalmanSerializer
from iqrp.app.regimes.kalman.smoothing import SmoothTrace, rts_smooth
from iqrp.app.regimes.kalman.trainer import KalmanTrainer, run_filter, simulate_lds
from iqrp.app.regimes.kalman.ukf import filter_ukf, sigma_points, unscented_transform

__all__ = [
    "FilterTrace",
    "KalmanDiagnostics",
    "KalmanEvaluator",
    "KalmanFilterModel",
    "KalmanRegimeModel",
    "KalmanSerializer",
    "KalmanSettings",
    "KalmanTrainer",
    "LinearGaussianSSM",
    "SmoothTrace",
    "adapt_noise_from_trace",
    "build_system",
    "filter_adaptive",
    "filter_ekf",
    "filter_linear",
    "filter_ukf",
    "rts_smooth",
    "run_filter",
    "sigma_points",
    "simulate_lds",
    "unscented_transform",
]
