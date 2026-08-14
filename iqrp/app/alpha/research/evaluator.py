"""Evaluate predictive power bundle for alpha candidates.

CRITICAL RULES (enforced in report warnings):
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Must track economic hypothesis on SignalDefinition.
- Point-in-time: signal helpers use only past windows; targets may use future.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_result import (
    SignalPerformance,
    SignalResearchReport,
    SignalScore,
    SignalStatistics,
    SignalStatus,
)
from iqrp.app.alpha.research.decay import analyze_decay, forward_returns
from iqrp.app.alpha.research.hit_rate import compute_hit_rate
from iqrp.app.alpha.research.information_coefficient import compute_ic, ic_summary
from iqrp.app.alpha.research.persistence import persistence_summary
from iqrp.app.alpha.research.predictor import SignalPredictor
from iqrp.app.alpha.research.rank_ic import compute_rank_ic
from iqrp.app.alpha.research.seasonality import analyze_seasonality
from iqrp.app.alpha.research.stability import analyze_stability


def compute_signal_statistics(values: np.ndarray) -> SignalStatistics:
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    n = len(x)
    n_fin = int(finite.size)
    if n_fin == 0:
        return SignalStatistics(
            n_obs=n,
            n_finite=0,
            mean=float("nan"),
            std=float("nan"),
            skew=float("nan"),
            kurtosis=float("nan"),
            min=float("nan"),
            max=float("nan"),
            missing_pct=100.0,
            autocorrelation_lag1=float("nan"),
        )
    ac1 = float("nan")
    if n_fin >= 3:
        from iqrp.app.alpha.research.persistence import autocorrelation

        ac1 = autocorrelation(x, 1)
    return SignalStatistics(
        n_obs=n,
        n_finite=n_fin,
        mean=float(np.mean(finite)),
        std=float(np.std(finite, ddof=1)) if n_fin > 1 else float("nan"),
        skew=float(stats.skew(finite)) if n_fin > 2 else float("nan"),
        kurtosis=float(stats.kurtosis(finite)) if n_fin > 3 else float("nan"),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        missing_pct=100.0 * (1.0 - n_fin / max(n, 1)),
        autocorrelation_lag1=ac1,
    )


def _hypothesis_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    # Crude length / keyword rubric — research triage only
    score = min(1.0, len(t) / 120.0)
    keywords = ("risk", "premium", "liquidity", "behavioral", "information", "flow", "inventory")
    hits = sum(1 for k in keywords if k in t.lower())
    score = min(1.0, score + 0.05 * hits)
    return float(score)


def _composite_score(
    performance: SignalPerformance,
    stability: dict[str, Any],
    persistence: dict[str, Any],
    hyp_score: float,
) -> SignalScore:
    pred = 0.0
    parts = 0
    if np.isfinite(performance.ic_mean):
        pred += min(1.0, abs(performance.ic_mean) / 0.1)
        parts += 1
    if np.isfinite(performance.hit_rate):
        pred += min(1.0, max(0.0, (performance.hit_rate - 0.5) / 0.2))
        parts += 1
    predictive = pred / max(parts, 1)

    stab = stability.get("stability_score", float("nan"))
    stability_s = float(stab) if np.isfinite(stab) else 0.0

    lag1 = persistence.get("lag1", float("nan"))
    persistence_s = float(min(1.0, abs(lag1))) if np.isfinite(lag1) else 0.0

    overall = float(
        0.40 * predictive + 0.25 * stability_s + 0.15 * persistence_s + 0.20 * hyp_score
    )
    return SignalScore(
        overall=overall * 100.0,
        predictive=predictive * 100.0,
        stability=stability_s * 100.0,
        persistence=persistence_s * 100.0,
        economic_hypothesis_score=hyp_score * 100.0,
        notes=(
            "Composite triage score only. "
            "Statistical significance alone ≠ alpha. "
            "Historical Sharpe alone cannot approve."
        ),
    )


class SignalEvaluator:
    """Bundle IC, rank IC, hit rate, decay, stability, persistence, seasonality."""

    def __init__(
        self,
        *,
        horizons: tuple[int, ...] = (1, 2, 5, 10),
        stability_window: int = 60,
        seasonality_period: int = 5,
    ) -> None:
        self.horizons = horizons
        self.stability_window = stability_window
        self.seasonality_period = seasonality_period

    def evaluate(
        self,
        signal: AlphaSignal | np.ndarray,
        returns: np.ndarray,
        *,
        definition: SignalDefinition | None = None,
        status: SignalStatus = SignalStatus.RESEARCHING,
    ) -> SignalResearchReport:
        if isinstance(signal, AlphaSignal):
            values = signal.values
            name = signal.name or (definition.name if definition else "signal")
            version = (
                definition.version
                if definition
                else str((signal.metadata or {}).get("definition", {}).get("version", "0.0.0"))
            )
            hyp = ""
            if definition is not None:
                hyp = definition.economic_hypothesis
            elif isinstance(signal.metadata.get("definition"), dict):
                hyp = str(signal.metadata["definition"].get("economic_hypothesis") or "")
                if definition is None:
                    try:
                        definition = SignalDefinition.from_dict(signal.metadata["definition"])
                        version = definition.version
                        name = definition.name
                        hyp = definition.economic_hypothesis
                    except Exception:
                        pass
        else:
            values = np.asarray(signal, dtype=np.float64)
            name = definition.name if definition else "signal"
            version = definition.version if definition else "0.0.0"
            hyp = definition.economic_hypothesis if definition else ""

        r = np.asarray(returns, dtype=np.float64)
        fwd1 = forward_returns(r, 1)
        ic = compute_ic(values, fwd1)
        ric = compute_rank_ic(values, fwd1)
        hit = compute_hit_rate(values, fwd1)
        ic_meta = ic_summary(values, fwd1, window=self.stability_window)
        decay = analyze_decay(values, r, horizons=self.horizons)
        stability = analyze_stability(values, r, horizon=1, window=self.stability_window)
        persistence = persistence_summary(values)
        seasonality = analyze_seasonality(values, r, period=self.seasonality_period, horizon=1)
        pred = SignalPredictor(horizon=1).predict(values, r)

        # Sharpe proxy from signed signal * forward return (research only)
        m = np.isfinite(values) & np.isfinite(fwd1)
        sharpe_proxy = float("nan")
        turnover_proxy = float("nan")
        if m.sum() > 5:
            pnl = np.sign(values[m]) * fwd1[m]
            mu, sd = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
            sharpe_proxy = mu / (sd + 1e-12) * np.sqrt(252.0) if sd > 0 else float("nan")
            sig_f = values[m]
            turnover_proxy = (
                float(np.mean(np.abs(np.diff(np.sign(sig_f))))) if len(sig_f) > 1 else float("nan")
            )

        performance = SignalPerformance(
            ic_mean=ic,
            ic_std=float(ic_meta.get("rolling_ic_std", float("nan"))),
            rank_ic_mean=ric,
            hit_rate=hit,
            predictive_r2=pred.r_squared,
            sharpe_proxy=sharpe_proxy,
            turnover_proxy=turnover_proxy,
            n_splits=pred.n_test,
            extras={
                "decay": decay,
                "rolling_ic_ir": ic_meta.get("rolling_ic_ir"),
            },
        )
        statistics = compute_signal_statistics(values)
        hyp_score = _hypothesis_score(hyp)
        score = _composite_score(performance, stability, persistence, hyp_score)

        warnings = [
            "Statistical significance alone ≠ alpha.",
            "Historical Sharpe alone cannot approve.",
            "Point-in-time: signal must use only past windows.",
        ]
        if not hyp.strip():
            warnings.append(
                "Missing economic_hypothesis — required on SignalDefinition before promotion."
            )
        if np.isfinite(sharpe_proxy):
            warnings.append(
                f"sharpe_proxy={sharpe_proxy:.3f} is diagnostic only and cannot approve."
            )

        return SignalResearchReport(
            signal_name=name,
            version=version,
            status=status,
            statistics=statistics,
            performance=performance,
            score=score,
            economic_hypothesis=hyp,
            diagnostics={
                "ic_summary": ic_meta,
                "decay": decay,
                "stability": stability,
                "persistence": persistence,
                "seasonality": seasonality,
                "prediction": pred.to_dict(),
            },
            warnings=warnings,
        )


def evaluate_signal(
    signal: AlphaSignal | np.ndarray,
    returns: np.ndarray,
    **kwargs: Any,
) -> SignalResearchReport:
    return SignalEvaluator().evaluate(signal, returns, **kwargs)
