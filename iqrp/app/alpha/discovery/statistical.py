"""Statistical screening for alpha *candidates* (correlation / IC).

CRITICAL:
- Results are flagged as CANDIDATES, NOT alpha.
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Screened features still require an economic_hypothesis before promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_result import SignalStatus
from iqrp.app.features.research._numeric import (
    information_coefficient,
    pearson,
    rank_information_coefficient,
    spearman,
)


@dataclass(slots=True)
class StatisticalCandidate:
    """A statistically screened feature flagged as research candidate only."""

    name: str
    feature: str
    ic: float
    rank_ic: float
    pearson: float
    spearman: float
    abs_ic: float
    is_alpha: bool = False  # ALWAYS False — statistical screen ≠ alpha
    status: SignalStatus = SignalStatus.CANDIDATE
    economic_hypothesis: str = (
        "Pending economic hypothesis. Statistical significance alone ≠ alpha."
    )
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature": self.feature,
            "ic": self.ic,
            "rank_ic": self.rank_ic,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "abs_ic": self.abs_ic,
            "is_alpha": self.is_alpha,
            "status": self.status.value,
            "economic_hypothesis": self.economic_hypothesis,
            "disclaimer": (
                "Flagged as CANDIDATE, NOT alpha. "
                "Statistical significance alone ≠ alpha. "
                "Historical Sharpe alone cannot approve."
            ),
            "extras": dict(self.extras),
        }

    def to_definition(
        self,
        *,
        lookback: int = 1,
        horizon: int = 1,
        owner: str = "research",
        universe: str = "default",
        frequency: str = "1d",
        economic_hypothesis: str | None = None,
    ) -> SignalDefinition:
        hyp = economic_hypothesis or self.economic_hypothesis
        return SignalDefinition(
            name=self.name,
            version="0.1.0",
            formula=f"screen({self.feature})",
            features=(self.feature,),
            lookback=lookback,
            horizon=horizon,
            universe=universe,
            frequency=frequency,
            direction="long_short",
            expected_relationship="positive" if self.ic >= 0 else "negative",
            economic_hypothesis=hyp,
            owner=owner,
            signal_type="statistical",
            parameters={
                "ic": self.ic,
                "rank_ic": self.rank_ic,
                "is_alpha": False,
            },
            tags=("statistical_screen", "candidate_not_alpha"),
            notes="Statistical screen only — not approved alpha.",
        )


def screen_features(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    min_abs_ic: float = 0.02,
    min_obs: int = 30,
    owner: str = "research",
) -> list[StatisticalCandidate]:
    """Corr / IC screen. Outputs are candidates — never labeled as alpha."""
    y = np.asarray(target, dtype=np.float64)
    out: list[StatisticalCandidate] = []
    for name, raw in features.items():
        x = np.asarray(raw, dtype=np.float64)
        if len(x) != len(y):
            raise ValueError(f"feature {name} length {len(x)} != target {len(y)}")
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < min_obs:
            continue
        ic = information_coefficient(x, y)
        ric = rank_information_coefficient(x, y)
        pr = pearson(x, y)
        sp = spearman(x, y)
        abs_ic = abs(ic) if np.isfinite(ic) else float("nan")
        if not np.isfinite(abs_ic) or abs_ic < min_abs_ic:
            continue
        out.append(
            StatisticalCandidate(
                name=f"stat_{name}",
                feature=name,
                ic=float(ic),
                rank_ic=float(ric),
                pearson=float(pr),
                spearman=float(sp),
                abs_ic=float(abs_ic),
                is_alpha=False,
                status=SignalStatus.CANDIDATE,
                extras={"owner": owner, "n_obs": int(mask.sum())},
            )
        )
    out.sort(key=lambda c: c.abs_ic, reverse=True)
    return out


def candidates_to_signals(
    candidates: list[StatisticalCandidate],
    features: dict[str, np.ndarray],
    *,
    economic_hypothesis: str | None = None,
) -> list[AlphaSignal]:
    """Materialize screened features as AlphaSignal candidates (not alpha)."""
    signals: list[AlphaSignal] = []
    for cand in candidates:
        if cand.feature not in features:
            continue
        definition = cand.to_definition(economic_hypothesis=economic_hypothesis)
        signals.append(
            AlphaSignal(
                values=np.asarray(features[cand.feature], dtype=np.float64),
                name=cand.name,
                definition_id=definition.definition_id,
                metadata={
                    "definition": definition.to_dict(),
                    "screen": cand.to_dict(),
                    "is_alpha": False,
                    "claims_profitability": False,
                },
            )
        )
    return signals
