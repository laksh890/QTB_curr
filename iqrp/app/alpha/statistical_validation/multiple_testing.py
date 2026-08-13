"""Multiple-testing adjustments with experiment accounting.

Wraps ``iqrp.app.timeseries.multiple_testing.adjust_pvalues`` when that module
is importable; otherwise uses an equivalent local implementation.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

Method = Literal["bonferroni", "holm", "fdr_bh", "none"]


def _local_adjust_pvalues(
    pvalues: np.ndarray | list[float],
    *,
    method: Method = "fdr_bh",
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


def _resolve_adjust_pvalues():
    try:
        from iqrp.app.timeseries.multiple_testing import adjust_pvalues as adj

        return adj
    except Exception:  # noqa: BLE001 — timeseries package may need optional deps
        return _local_adjust_pvalues


class ExperimentTracker:
    """Track cumulative hypothesis tests to avoid silent multiple testing."""

    def __init__(self) -> None:
        self.n_experiments: int = 0
        self.history: list[dict[str, Any]] = []

    def record(self, n: int, *, label: str | None = None, meta: dict[str, Any] | None = None) -> None:
        k = max(int(n), 0)
        self.n_experiments += k
        self.history.append(
            {
                "label": label,
                "n_added": k,
                "n_experiments": self.n_experiments,
                "meta": dict(meta or {}),
            }
        )

    def reset(self) -> None:
        self.n_experiments = 0
        self.history.clear()


_GLOBAL_TRACKER = ExperimentTracker()


def get_experiment_tracker() -> ExperimentTracker:
    return _GLOBAL_TRACKER


def multiple_testing_adjustment(
    pvalues: np.ndarray | list[float],
    *,
    method: Method = "fdr_bh",
    alpha: float = 0.05,
    tracker: ExperimentTracker | None = None,
    label: str | None = None,
    record: bool = True,
) -> dict[str, Any]:
    """Adjust p-values and optionally increment the experiment counter.

    Returns the ``adjust_pvalues`` payload plus ``n_experiments`` (session total
    after this call when ``record=True``).
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    adjust_pvalues = _resolve_adjust_pvalues()
    result = dict(adjust_pvalues(p, method=method, alpha=alpha))
    tr = tracker if tracker is not None else _GLOBAL_TRACKER
    if record:
        tr.record(int(p.size), label=label, meta={"method": method, "alpha": float(alpha)})
    result["n_experiments"] = int(tr.n_experiments)
    result["n_tests_this_call"] = int(p.size)
    result["raw_pvalues"] = p.tolist()
    return result
