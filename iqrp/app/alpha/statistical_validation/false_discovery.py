"""False discovery rate utilities for alpha research batteries."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.alpha.statistical_validation.multiple_testing import (
    ExperimentTracker,
    multiple_testing_adjustment,
)

Method = Literal["bonferroni", "holm", "fdr_bh", "none"]


def storey_qvalues(
    pvalues: np.ndarray | list[float],
    *,
    lambda_: float = 0.5,
) -> dict[str, Any]:
    """Storey (2002) π₀ estimate and q-values (conservative, sorted)."""
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, 1.0)
    m = int(p.size)
    if m == 0:
        return {"qvalues": np.asarray([], dtype=np.float64), "pi0": float("nan"), "m": 0}

    lam = float(np.clip(lambda_, 0.0, 0.95))
    pi0 = float(np.mean(p > lam) / max(1.0 - lam, 1e-12))
    pi0 = float(np.clip(pi0, 0.0, 1.0))

    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(m, dtype=np.float64)
    # q_(i) = min_{k>=i} (π0 m p_(k) / k)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = pi0 * m * ranked[i] / float(i + 1)
        prev = min(prev, val)
        q[i] = prev
    qvalues = np.empty(m, dtype=np.float64)
    qvalues[order] = np.clip(q, 0.0, 1.0)
    return {"qvalues": qvalues, "pi0": pi0, "m": m, "lambda": lam}


def false_discovery_report(
    pvalues: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
    method: Method = "fdr_bh",
    lambda_: float = 0.5,
    tracker: ExperimentTracker | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Combined BH adjustment + Storey q-values + discovery counts."""
    adj = multiple_testing_adjustment(
        pvalues,
        method=method,
        alpha=alpha,
        tracker=tracker,
        label=label or "false_discovery",
        record=True,
    )
    storey = storey_qvalues(pvalues, lambda_=lambda_)
    rejected = np.asarray(adj["rejected"], dtype=bool)
    return {
        "method": method,
        "alpha": float(alpha),
        "adjusted": adj["adjusted"],
        "rejected": rejected,
        "n_discoveries": int(np.sum(rejected)),
        "n_tests": int(np.asarray(pvalues).size),
        "n_experiments": adj["n_experiments"],
        "qvalues": storey["qvalues"],
        "pi0": storey["pi0"],
        "fdr_estimate": (
            float(storey["pi0"] * float(alpha)) if np.isfinite(storey["pi0"]) else float("nan")
        ),
    }
