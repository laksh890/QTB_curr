"""Transformer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TransformerDiagnosticReport:
    history: dict[str, Any]
    residual_mean: float
    residual_std: float
    residual_skew: float
    attention_entropy: float
    embedding_drift: float
    weight_stats: dict[str, float]
    calibration: dict[str, list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": dict(self.history),
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "residual_skew": self.residual_skew,
            "attention_entropy": self.attention_entropy,
            "embedding_drift": self.embedding_drift,
            "weight_stats": dict(self.weight_stats),
            "calibration": {k: list(v) for k, v in self.calibration.items()},
            "metadata": dict(self.metadata),
        }


def run_transformer_diagnostics(model: Any) -> TransformerDiagnosticReport:
    resid = np.asarray(model._residuals if getattr(model, "_residuals", None) is not None else [0.0], dtype=np.float64)
    if resid.size == 0:
        resid = np.asarray([0.0], dtype=np.float64)
    mean = float(np.mean(resid))
    std = float(np.std(resid))
    skew = float(np.mean(((resid - mean) / std) ** 3)) if std > 1e-12 else 0.0
    hist = model._history.to_dict() if hasattr(model, "_history") else {}
    attn_ent = _attention_entropy(getattr(model, "_last_attn", None))
    emb_drift = _embedding_drift(getattr(model, "_last_embeddings", None), getattr(model, "_X_seq", None))
    wstats = _weight_stats(getattr(model, "_module", None))
    cal = _calibration(resid)
    return TransformerDiagnosticReport(
        history=hist,
        residual_mean=mean,
        residual_std=std,
        residual_skew=skew,
        attention_entropy=attn_ent,
        embedding_drift=emb_drift,
        weight_stats=wstats,
        calibration=cal,
        metadata={"architecture": getattr(model, "architecture_name", "transformer")},
    )


def _attention_entropy(attn: Any) -> float:
    if attn is None:
        return 0.0
    a = np.asarray(attn, dtype=np.float64)
    a = np.clip(a, 1e-8, 1.0)
    a = a / a.sum(axis=-1, keepdims=True)
    ent = -np.sum(a * np.log(a), axis=-1)
    return float(np.mean(ent))


def _embedding_drift(emb: Any, X_seq: Any) -> float:
    if emb is None or X_seq is None:
        return 0.0
    e = np.asarray(emb, dtype=np.float64).reshape(-1)
    x = np.asarray(X_seq[-1:], dtype=np.float64).reshape(-1)
    n = min(e.size, x.size)
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(e[:n] - x[:n])))


def _weight_stats(module: Any) -> dict[str, float]:
    if module is None or not hasattr(module, "parameters"):
        return {}
    vals = [p.detach().cpu().numpy().reshape(-1) for p in module.parameters()]
    if not vals:
        return {}
    w = np.concatenate(vals)
    if w.size == 0:
        return {}
    return {"mean": float(np.mean(w)), "std": float(np.std(w)), "abs_max": float(np.max(np.abs(w))), "n": float(w.size)}


def _calibration(resid: np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    if resid.size == 0:
        return {"levels": [], "coverage": []}
    levels = np.linspace(0.1, 0.9, n_bins)
    coverage = [float(np.mean(np.abs(resid) <= np.quantile(np.abs(resid), lv))) for lv in levels]
    return {"levels": levels.tolist(), "coverage": coverage}
