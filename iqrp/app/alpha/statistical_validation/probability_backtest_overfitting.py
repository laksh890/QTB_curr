"""Probability of Backtest Overfitting (PBO) via CSCV approximation.

Implements a simplified Combinatorially Symmetric Cross-Validation (CSCV)
estimate of PBO (Bailey, Borwein, López de Prado, Zhu).

Look-ahead prevention
---------------------
Folds are contiguous time blocks. Combinations assign whole blocks to IS/OOS
without shuffling observations across time, preserving temporal structure
inside each block.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def _sharpe(r: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd < 1e-15:
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def _as_strategy_matrix(returns: Any) -> np.ndarray:
    """Return shape (T, N) — columns are strategy return paths."""
    a = np.asarray(returns, dtype=np.float64)
    if a.ndim == 1:
        # Single path: build synthetic rival strategies via sign/scale variants
        # so CSCV is well-defined for a lone series smoke path.
        r = a.reshape(-1, 1)
        rivals = np.column_stack(
            [
                r[:, 0],
                -r[:, 0],
                np.roll(r[:, 0], 1),
                np.maximum(r[:, 0], 0.0),
                -np.maximum(r[:, 0], 0.0),
            ]
        )
        rivals[0, 2] = r[0, 0]
        return rivals
    if a.ndim == 2:
        # Prefer (T, N); if N >> T assume (N, T) and transpose
        if a.shape[0] < a.shape[1] and a.shape[0] < 16:
            return a.T
        return a
    raise ValueError("returns must be 1-D or 2-D array")


def probability_backtest_overfitting(
    returns: Any,
    *,
    n_groups: int = 8,
    periods_per_year: float = 252.0,
    max_combinations: int = 2000,
    metric: str = "sharpe",
) -> dict[str, Any]:
    """Estimate PBO with CSCV over contiguous OOS folds.

    Parameters
    ----------
    returns:
        ``(T,)`` single return series or ``(T, N)`` matrix of N strategy
        return paths aligned in time.
    n_groups:
        Even number of contiguous time groups (CSCV partitions).
    max_combinations:
        Cap on C(S, S/2) evaluated (uniform subsample if larger).

    Returns
    -------
    dict with ``pbo``, ``n_combinations``, ``logit_lambda``, etc.
    """
    mat = _as_strategy_matrix(returns)
    t, n_strat = mat.shape
    s = int(n_groups)
    if s < 2:
        s = 2
    if s % 2 == 1:
        s += 1
    # Trim to divisible length
    block = t // s
    if block < 2 or n_strat < 2:
        return {
            "pbo": float("nan"),
            "n_combinations": 0,
            "n_strategies": int(n_strat),
            "n_groups": s,
            "method": "cscv",
            "detail": "insufficient_data",
        }
    mat = mat[: block * s]
    groups = [mat[i * block : (i + 1) * block] for i in range(s)]

    half = s // 2
    all_combos = list(combinations(range(s), half))
    if len(all_combos) > int(max_combinations):
        rng = np.random.default_rng(0)
        pick = rng.choice(len(all_combos), size=int(max_combinations), replace=False)
        all_combos = [all_combos[i] for i in pick]

    overfit = 0
    logits: list[float] = []
    for is_idx in all_combos:
        is_set = set(is_idx)
        oos_idx = [i for i in range(s) if i not in is_set]
        is_ret = np.concatenate([groups[i] for i in is_idx], axis=0)
        oos_ret = np.concatenate([groups[i] for i in oos_idx], axis=0)

        if metric == "mean":
            is_perf = np.nanmean(is_ret, axis=0)
            oos_perf = np.nanmean(oos_ret, axis=0)
        else:
            is_perf = np.array(
                [_sharpe(is_ret[:, j], periods_per_year=periods_per_year) for j in range(n_strat)]
            )
            oos_perf = np.array(
                [_sharpe(oos_ret[:, j], periods_per_year=periods_per_year) for j in range(n_strat)]
            )

        if not np.any(np.isfinite(is_perf)):
            continue
        best = int(np.nanargmax(is_perf))
        oos_best = float(oos_perf[best])
        med = float(np.nanmedian(oos_perf))
        if not np.isfinite(oos_best) or not np.isfinite(med):
            continue
        # Relative rank of best-IS strategy among OOS performances
        rank = float(np.mean(oos_perf <= oos_best))  # higher better
        # λ = logit of relative rank; PBO counts cases where OOS < median
        if oos_best < med:
            overfit += 1
        # avoid 0/1 for logit
        r = float(np.clip(rank, 1e-6, 1.0 - 1e-6))
        logits.append(float(np.log(r / (1.0 - r))))

    n_comb = max(len(logits), 1)
    pbo = float(overfit / n_comb) if logits else float("nan")
    return {
        "pbo": pbo,
        "n_combinations": int(len(logits)),
        "n_overfit": int(overfit),
        "n_strategies": int(n_strat),
        "n_groups": s,
        "block_size": int(block),
        "logit_lambda_mean": float(np.mean(logits)) if logits else float("nan"),
        "method": "cscv",
        "metric": metric,
    }


# Short alias
pbo_cscv = probability_backtest_overfitting
