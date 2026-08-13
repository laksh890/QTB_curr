"""Model disagreement and consensus metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import entropy, js_divergence


def pairwise_disagreement(member_proba: list[np.ndarray]) -> np.ndarray:
    """Mean JS divergence between all member pairs at each time — shape ``(T,)``."""
    m = len(member_proba)
    t = member_proba[0].shape[0]
    if m < 2:
        return np.zeros(t, dtype=np.float64)
    out = np.zeros(t, dtype=np.float64)
    pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            for tt in range(t):
                out[tt] += float(js_divergence(member_proba[i][tt], member_proba[j][tt]))
            pairs += 1
    return out / max(pairs, 1)


def consensus_score(member_proba: list[np.ndarray]) -> np.ndarray:
    """1 - normalized disagreement (clipped)."""
    d = pairwise_disagreement(member_proba)
    # JS is in [0, ln2] approx for binary; normalize softly
    return np.clip(1.0 - d / np.log(2.0), 0.0, 1.0)


def hard_agreement(member_proba: list[np.ndarray]) -> np.ndarray:
    """Fraction of members agreeing with majority hard label — ``(T,)``."""
    m = len(member_proba)
    t = member_proba[0].shape[0]
    k = member_proba[0].shape[1]
    out = np.zeros(t, dtype=np.float64)
    for tt in range(t):
        votes = np.zeros(k, dtype=np.float64)
        for p in member_proba:
            votes[int(np.argmax(p[tt]))] += 1.0
        out[tt] = float(np.max(votes) / max(m, 1))
    return out


def prediction_diversity(member_proba: list[np.ndarray]) -> float:
    """Fraction of unique hard label sequences across members (summary)."""
    hards = [tuple(np.argmax(p, axis=1).tolist()) for p in member_proba]
    return float(len(set(hards)) / max(len(hards), 1))


def mean_entropy(member_proba: list[np.ndarray]) -> np.ndarray:
    t = member_proba[0].shape[0]
    out = np.zeros(t, dtype=np.float64)
    for tt in range(t):
        out[tt] = float(np.mean([entropy(p[tt]) for p in member_proba]))
    return out


def disagreement_report(
    member_proba: list[np.ndarray],
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    cons = consensus_score(member_proba)
    agree = hard_agreement(member_proba)
    diss = pairwise_disagreement(member_proba)
    return {
        "mean_disagreement": float(np.mean(diss)),
        "mean_consensus": float(np.mean(cons)),
        "mean_agreement": float(np.mean(agree)),
        "prediction_diversity": prediction_diversity(member_proba),
        "mean_entropy": float(np.mean(mean_entropy(member_proba))),
        "consensus_timeline": cons,
        "agreement_timeline": agree,
        "disagreement_timeline": diss,
        "member_names": list(names or []),
    }
