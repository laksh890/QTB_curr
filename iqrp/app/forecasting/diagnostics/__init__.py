"""Forecast model diagnostics."""

from iqrp.app.forecasting.diagnostics.calibration import CalibrationReport, calibration_report, detect_bias
from iqrp.app.forecasting.diagnostics.drift import (
    DriftReport,
    detect_feature_drift,
    detect_prediction_drift,
    ks_statistic,
    mean_shift,
    psi,
)
from iqrp.app.forecasting.diagnostics.residuals import (
    ResidualReport,
    autocorrelation,
    compute_residuals,
    forecast_error_by_horizon,
    residual_analysis,
)

__all__ = [
    "CalibrationReport",
    "DriftReport",
    "ResidualReport",
    "autocorrelation",
    "calibration_report",
    "compute_residuals",
    "detect_bias",
    "detect_feature_drift",
    "detect_prediction_drift",
    "forecast_error_by_horizon",
    "ks_statistic",
    "mean_shift",
    "psi",
    "residual_analysis",
]
