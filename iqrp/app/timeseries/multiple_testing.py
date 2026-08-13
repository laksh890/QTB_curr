"""Multiple-testing adjustments for large-scale time-series research.

Never treat an unadjusted p-value as evidence that a feature is profitable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def adjust_pvalues(
    pvalues: np.ndarray | list[float],
    *,
    method: Literal["bonferroni", "holm", "fdr_bh", "none"] = "fdr_bh",
    alpha: float = 0.05,
) -> dict[str, np.ndarray | float | str]:
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, 1.0)
    m = p.size
    if m == 0 or method == "none":
        return {"adjusted": p.copy(), "rejected": (p < alpha).astype(bool), "method": method, "alpha": alpha}

    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(m, dtype=np.float64)

    if method == "bonferroni":
        adjusted[order] = np.minimum(ranked * m, 1.0)
    elif method == "holm":
        holm = np.minimum(ranked * (m - np.arange(m)), 1.0)
        # enforce monotonicity
        for i in range(m - 2, -1, -1):
            holm[i] = min(holm[i], holm[i + 1])
        adjusted[order] = holm
    else:  # fdr_bh
        ranks = np.arange(1, m + 1, dtype=np.float64)
        bh = ranked * m / ranks
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        adjusted[order] = np.minimum(bh, 1.0)

    return {
        "adjusted": adjusted,
        "rejected": adjusted < alpha,
        "method": method,
        "alpha": float(alpha),
        "n_tests": float(m),
    }
