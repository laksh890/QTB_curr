"""Model evaluation and leaderboard for ensemble members."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import cross_entropy
from iqrp.app.regimes.ensemble.calibration import brier_score, expected_calibration_error


class EnsembleEvaluator:
    def evaluate_member(
        self,
        *,
        proba: np.ndarray,
        hard: np.ndarray,
        truth: np.ndarray | None,
        log_likelihood: float = 0.0,
    ) -> dict[str, float]:
        p = np.asarray(proba, dtype=np.float64)
        h = np.asarray(hard, dtype=np.int64).reshape(-1)
        metrics: dict[str, float] = {
            "log_likelihood": float(log_likelihood),
            "mean_max_proba": float(np.mean(p.max(axis=1))) if p.size else 0.0,
        }
        if truth is not None:
            y = np.asarray(truth, dtype=np.int64).reshape(-1)
            n = min(h.size, y.size, p.shape[0])
            metrics["accuracy"] = float(np.mean(h[:n] == y[:n]))
            metrics["cross_entropy"] = float(
                np.mean([cross_entropy(_onehot(y[i], p.shape[1]), p[i]) for i in range(n)])
            )
            metrics["calibration_error"] = expected_calibration_error(p[:n], y[:n])
            metrics["brier"] = brier_score(p[:n], y[:n])
            # drawdown-aware: accuracy on worst quartile of confidence
            conf = p[:n].max(axis=1)
            q = np.quantile(conf, 0.25) if n else 0.0
            mask = conf <= q
            if np.any(mask):
                metrics["drawdown_accuracy"] = float(np.mean(h[:n][mask] == y[:n][mask]))
            else:
                metrics["drawdown_accuracy"] = metrics["accuracy"]
        return metrics

    def leaderboard(
        self,
        *,
        member_probas: dict[str, np.ndarray],
        member_hards: dict[str, np.ndarray],
        log_likes: dict[str, float],
        truth: np.ndarray | None = None,
        ensemble_proba: np.ndarray | None = None,
        ensemble_hard: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, p in member_probas.items():
            m = self.evaluate_member(
                proba=p,
                hard=member_hards[name],
                truth=truth,
                log_likelihood=log_likes.get(name, 0.0),
            )
            m["name"] = name
            rows.append(m)
        if ensemble_proba is not None and ensemble_hard is not None:
            em = self.evaluate_member(
                proba=ensemble_proba,
                hard=ensemble_hard,
                truth=truth,
                log_likelihood=float(
                    np.sum(np.log(np.clip(ensemble_proba.max(axis=1), 1e-300, None)))
                ),
            )
            em["name"] = "ensemble"
            rows.append(em)
        # rank by accuracy if available else log_likelihood
        key = "accuracy" if truth is not None else "log_likelihood"
        rows.sort(key=lambda r: float(r.get(key, -1e9)), reverse=True)
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return rows


def _onehot(label: int, k: int) -> np.ndarray:
    v = np.zeros(k, dtype=np.float64)
    if 0 <= int(label) < k:
        v[int(label)] = 1.0
    return v
