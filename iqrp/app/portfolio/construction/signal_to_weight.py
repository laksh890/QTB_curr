"""Map alpha / forecast signals to raw portfolio weights.

This module does **not** generate alpha — it only expresses provided signals
as unconstrained (or lightly normalized) raw weights via rank, z-score, or softmax.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

SignalMethod = Literal["rank", "zscore", "softmax", "proportional", "identity"]


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def signals_to_raw_weights(
    signals: Sequence[float] | np.ndarray,
    *,
    method: SignalMethod | str = "zscore",
    long_only: bool = True,
    temperature: float = 1.0,
    budget: float = 1.0,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Express signals as raw weights (pre-constraint / pre-optimization).

    Parameters
    ----------
    signals:
        Asset-level alpha / forecast scores (already computed elsewhere).
    method:
        ``rank`` | ``zscore`` | ``softmax`` | ``proportional`` | ``identity``.
    long_only:
        If True, negative raw scores are clipped or shifted before normalization.
    temperature:
        Softmax temperature (higher → flatter). Also scales z-score intensity.
    budget:
        Target sum of absolute (long-only) or net (long-short) weights.
    """
    s = _as_1d(signals)
    n = int(s.size)
    if n == 0:
        return {
            "name": "signals_to_raw_weights",
            "method": str(method),
            "weights": [],
            "raw": [],
            "names": list(names) if names is not None else [],
            "budget": float(budget),
            "long_only": bool(long_only),
        }

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    m = str(method).strip().lower()
    temp = max(float(temperature), 1e-8)

    if m == "identity":
        raw = s.copy()
    elif m == "rank":
        order = np.argsort(np.argsort(s))
        raw = (order.astype(np.float64) + 1.0) / float(n)
        raw = raw - float(np.mean(raw))
    elif m == "zscore":
        mu = float(np.mean(s))
        sd = float(np.std(s))
        raw = (s - mu) / (sd + 1e-12) / temp
    elif m == "softmax":
        logits = s / temp
        logits = logits - float(np.max(logits))
        ex = np.exp(logits)
        raw = ex / max(float(np.sum(ex)), 1e-18)
    elif m in ("proportional", "prop"):
        raw = s.copy()
    else:
        raise ValueError(f"Unknown signal method '{method}'. Use rank|zscore|softmax|proportional|identity")

    if m == "softmax":
        w = raw * float(budget)
    elif long_only:
        pos = np.maximum(raw, 0.0)
        if m == "rank":
            # ranks shifted around 0 → lift to positive
            pos = raw - float(np.min(raw)) + 1e-12
        if float(np.sum(pos)) <= 1e-18:
            w = np.full(n, float(budget) / n, dtype=np.float64)
        else:
            w = pos / float(np.sum(pos)) * float(budget)
    else:
        # dollar-neutral-ish: demean then scale L1 to budget*2 (gross) or net to budget
        demeaned = raw - float(np.mean(raw))
        gross = float(np.sum(np.abs(demeaned)))
        if gross <= 1e-18:
            w = np.zeros(n, dtype=np.float64)
        else:
            # net exposure ≈ 0; scale so gross ≈ |budget| when budget is gross target
            target_gross = abs(float(budget)) if abs(float(budget)) > 1e-12 else 1.0
            w = demeaned * (target_gross / gross)

    name_list = list(names) if names is not None else [f"a{i}" for i in range(n)]
    if len(name_list) != n:
        name_list = [f"a{i}" for i in range(n)]

    return {
        "name": "signals_to_raw_weights",
        "method": m,
        "weights": [float(x) for x in w.tolist()],
        "raw": [float(x) for x in raw.tolist()],
        "names": name_list,
        "budget": float(budget),
        "long_only": bool(long_only),
        "temperature": float(temp),
        "signal_sum": float(np.sum(s)),
        "weight_sum": float(np.sum(w)),
        "gross": float(np.sum(np.abs(w))),
    }


def rank_weights(signals: Any, **kwargs: Any) -> dict[str, Any]:
    return signals_to_raw_weights(signals, method="rank", **kwargs)


def zscore_weights(signals: Any, **kwargs: Any) -> dict[str, Any]:
    return signals_to_raw_weights(signals, method="zscore", **kwargs)


def softmax_weights(signals: Any, **kwargs: Any) -> dict[str, Any]:
    return signals_to_raw_weights(signals, method="softmax", **kwargs)
