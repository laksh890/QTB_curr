"""Self-diagnostics for the ensemble regime engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.ensemble.calibration import expected_calibration_error
from iqrp.app.regimes.ensemble.disagreement import disagreement_report
from iqrp.app.regimes.ensemble.registry import EnsembleMember


class EnsembleDiagnostics:
    def report(
        self,
        *,
        members: list[EnsembleMember],
        weights: np.ndarray,
        ensemble_proba: np.ndarray,
        member_probas: list[np.ndarray],
        names: list[str],
        history: list[dict[str, Any]] | None = None,
        truth: np.ndarray | None = None,
        leaderboard: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        diss = disagreement_report(member_probas, names=names)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        weight_map = {n: float(wi) for n, wi in zip(names, w, strict=False)}
        failures = [
            {"name": m.name, "error": m.metadata.get("error")}
            for m in members
            if m.metadata.get("error")
        ]
        # simple drift: mean absolute weight change across history
        drift = 0.0
        if history and len(history) >= 2:
            w0 = history[0].get("weights") or {}
            w1 = history[-1].get("weights") or {}
            keys = set(w0) | set(w1)
            if keys:
                drift = float(
                    np.mean([abs(float(w1.get(k, 0)) - float(w0.get(k, 0))) for k in keys])
                )
        cal = None
        if truth is not None:
            y = np.asarray(truth, dtype=np.int64).reshape(-1)
            cal = expected_calibration_error(ensemble_proba, y[: ensemble_proba.shape[0]])
        return {
            "leaderboard": leaderboard or [],
            "weights": weight_map,
            "weight_evolution": [h.get("weights") for h in (history or [])],
            "disagreement": {
                "mean_disagreement": diss["mean_disagreement"],
                "mean_consensus": diss["mean_consensus"],
                "mean_agreement": diss["mean_agreement"],
                "prediction_diversity": diss["prediction_diversity"],
            },
            "agreement_matrix": _agreement_matrix(member_probas, names),
            "calibration": {"ece": cal, "enabled": cal is not None},
            "confidence": {
                "mean_max_proba": float(np.mean(ensemble_proba.max(axis=1))),
                "mean_consensus": diss["mean_consensus"],
            },
            "failures": failures,
            "drift": {"weight_drift": drift},
            "n_members": len(names),
            "n_obs": int(ensemble_proba.shape[0]),
        }


def _agreement_matrix(member_probas: list[np.ndarray], names: list[str]) -> dict[str, Any]:
    m = len(member_probas)
    mat = np.eye(m)
    for i in range(m):
        hi = np.argmax(member_probas[i], axis=1)
        for j in range(i + 1, m):
            hj = np.argmax(member_probas[j], axis=1)
            n = min(hi.size, hj.size)
            a = float(np.mean(hi[:n] == hj[:n])) if n else 0.0
            mat[i, j] = mat[j, i] = a
    return {"names": list(names), "matrix": mat.tolist()}
