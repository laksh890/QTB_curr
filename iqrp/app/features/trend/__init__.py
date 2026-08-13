"""Trend category exports (registration side-effect)."""

from iqrp.app.features.trend.features import (
    DistanceFromHigh,
    DistanceFromLow,
    EmaSlope,
    LinearRegressionSlope,
    LogReturn,
    MultiPeriodReturn,
    PriceAcceleration,
    PriceCurvature,
    RollingHigh,
    RollingLow,
    RollingTrend,
    SmaSlope,
    TrendStrength,
)

__all__ = [
    "DistanceFromHigh",
    "DistanceFromLow",
    "EmaSlope",
    "LinearRegressionSlope",
    "LogReturn",
    "MultiPeriodReturn",
    "PriceAcceleration",
    "PriceCurvature",
    "RollingHigh",
    "RollingLow",
    "RollingTrend",
    "SmaSlope",
    "TrendStrength",
]
