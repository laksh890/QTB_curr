"""Neural diagnostics report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class NeuralDiagnosticReport:
    history: dict[str, Any]
    residual_mean: float
    residual_std: float
    residual_skew: float
    grad_norm_mean: float
    weight_stats: dict[str, float]
    activation_stats: dict[str, float]
    calibration: dict[str, list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": dict(self.history),
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "residual_skew": self.residual_skew,
            "grad_norm_mean": self.grad_norm_mean,
            "weight_stats": dict(self.weight_stats),
            "activation_stats": dict(self.activation_stats),
            "calibration": {k: list(v) for k, v in self.calibration.items()},
            "metadata": dict(self.metadata),
        }


def run_neural_diagnostics(model: Any) -> NeuralDiagnosticReport:
    resid = np.asarray(model._residuals if model._residuals is not None else [0.0], dtype=np.float64)
    if resid.size == 0:
        resid = np.asarray([0.0], dtype=np.float64)
    mean = float(np.mean(resid))
    std = float(np.std(resid))
    skew = float(np.mean(((resid - mean) / std) ** 3)) if std > 1e-12 else 0.0
    hist = model._history.to_dict() if hasattr(model, "_history") else {}
    gnorms = hist.get("grad_norms") or [0.0]
    weight_stats = _weight_stats(getattr(model, "_module", None))
    act_stats = _activation_stats(model)
    cal = _calibration_curve(resid)
    return NeuralDiagnosticReport(
        history=hist,
        residual_mean=mean,
        residual_std=std,
        residual_skew=skew,
        grad_norm_mean=float(np.mean(gnorms)),
        weight_stats=weight_stats,
        activation_stats=act_stats,
        calibration=cal,
        metadata={"architecture": getattr(model, "architecture_name", "neural")},
    )


def _weight_stats(module: Any) -> dict[str, float]:
    if module is None or not hasattr(module, "parameters"):
        return {}
    vals = []
    for p in module.parameters():
        vals.append(p.detach().cpu().numpy().reshape(-1))
    if not vals:
        return {}
    w = np.concatenate(vals)
    if w.size == 0:
        return {}
    return {"mean": float(np.mean(w)), "std": float(np.std(w)), "abs_max": float(np.max(np.abs(w))), "n": float(w.size)}


def _activation_stats(model: Any) -> dict[str, float]:
    if getattr(model, "_X_seq", None) is None or getattr(model, "_module", None) is None:
        return {}
    try:
        from iqrp.app.forecasting.neural.base.trainer import NeuralTrainer

        trainer = NeuralTrainer(model._neural_settings)
        trainer.device = model._device
        out = trainer.predict(model._module, model._X_seq[:32])
        return {"pred_mean": float(np.mean(out)), "pred_std": float(np.std(out))}
    except Exception:  # noqa: BLE001  # pragma: no cover
        return {}


def _calibration_curve(resid: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    r = np.sort(np.abs(resid))
    if r.size == 0:
        return {"levels": [], "coverage": []}
    levels = np.linspace(0.1, 0.9, n_bins)
    coverage = [float(np.mean(np.abs(resid) <= np.quantile(np.abs(resid), lv))) for lv in levels]
    return {"levels": levels.tolist(), "coverage": coverage}
