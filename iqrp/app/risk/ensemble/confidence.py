"""Ensemble confidence estimation.

Forecast confidence must never override hard risk limits — this module only
produces a soft confidence score for reporting and soft sizing within caps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure
from iqrp.app.risk.ensemble.config import EnsembleSettings


def estimate_confidence(
    metrics: dict[str, Any],
    *,
    settings: EnsembleSettings,
    disagreement: float | None = None,
    sample_size: int | None = None,
    missing_critical_count: int = 0,
    data_quality: float | None = None,
    model_stability: float | None = None,
    as_measure: bool = False,
) -> float | RiskMeasure:
    """Compute ensemble confidence in [min, max].

    Lower when disagreement is high, sample size is thin, or critical metrics missing.
    """
    cfg = settings.confidence
    conf = float(cfg.base)

    # Sample-size factor
    if sample_size is not None:
        n = max(int(sample_size), 0)
        floor = max(int(cfg.sample_size_floor), 1)
        full = max(int(cfg.sample_size_full), floor)
        if n < floor:
            sample_factor = 0.35 * (n / floor)
        else:
            sample_factor = 0.35 + 0.65 * min((n - floor) / max(full - floor, 1), 1.0)
        conf *= float(np.clip(sample_factor, 0.05, 1.0))

    # Disagreement penalty
    disc = disagreement
    if disc is None and "disagreement" in metrics:
        dval = metrics["disagreement"]
        if isinstance(dval, dict):
            disc = float(dval.get("overall_disagreement", 0.0) or 0.0)
        else:
            try:
                disc = float(dval)
            except (TypeError, ValueError):
                disc = None
    if disc is not None:
        conf *= float(np.clip(1.0 - float(cfg.disagreement_penalty) * float(disc), 0.05, 1.0))

    # Missing critical metrics — confidence collapses; never implies approval
    if missing_critical_count > 0:
        conf *= float(
            np.clip(1.0 - float(cfg.missing_metric_penalty) * float(missing_critical_count), 0.05, 1.0)
        )

    if data_quality is not None:
        conf *= float(np.clip(data_quality, 0.0, 1.0))
    if model_stability is not None:
        conf *= float(np.clip(model_stability, 0.0, 1.0))

    # Metric coverage soft signal
    numeric_keys = [
        k
        for k, v in metrics.items()
        if not str(k).startswith("_")
        and (
            isinstance(v, (int, float))
            or (isinstance(v, dict) and "value" in v)
        )
    ]
    coverage = min(len(numeric_keys) / max(len(settings.critical_metric_keys), 1), 1.0)
    conf *= 0.7 + 0.3 * coverage

    conf = float(np.clip(conf, float(cfg.min_confidence), float(cfg.max_confidence)))

    if as_measure:
        return RiskMeasure(
            name="ensemble_confidence",
            value=conf,
            unit="probability",
            confidence=conf,
            method="ensemble_confidence",
            parameters={
                "disagreement": disc,
                "sample_size": sample_size,
                "missing_critical_count": int(missing_critical_count),
                "data_quality": data_quality,
                "model_stability": model_stability,
                "note": "Soft signal only; cannot override hard risk limits",
            },
        )
    return conf


class ConfidenceEstimator:
    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings

    def estimate(self, metrics: dict[str, Any], **kwargs: Any) -> float | RiskMeasure:
        return estimate_confidence(metrics, settings=self.settings, **kwargs)
