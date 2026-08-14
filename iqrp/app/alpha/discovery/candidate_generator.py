"""Orchestrate alpha-candidate discovery from features, forecasts, and formulas.

CRITICAL:
- Discovery emits CANDIDATES, not approved alpha.
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Every produced SignalDefinition must carry an economic_hypothesis.
- Signal helpers are point-in-time (no future leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_registry import SignalRegistry, get_default_registry
from iqrp.app.alpha.base.signal_result import SignalStatus
from iqrp.app.alpha.discovery import (
    alternative as alt_mod,
    cross_sectional as cs_mod,
    event_based as event_mod,
    statistical as stat_mod,
    symbolic as sym_mod,
    time_series as ts_mod,
)


@dataclass(slots=True)
class DiscoveryResult:
    """Bundle of discovered candidates (explicitly not alpha)."""

    signals: list[AlphaSignal] = field(default_factory=list)
    definitions: list[SignalDefinition] = field(default_factory=list)
    statistical_screens: list[stat_mod.StatisticalCandidate] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_signals": len(self.signals),
            "n_definitions": len(self.definitions),
            "n_screens": len(self.statistical_screens),
            "experiment_ids": list(self.experiment_ids),
            "notes": list(self.notes),
            "disclaimer": (
                "Discovery outputs are CANDIDATES, NOT alpha. "
                "Statistical significance alone ≠ alpha. "
                "Historical Sharpe alone cannot approve."
            ),
        }


class CandidateGenerator:
    """Orchestrate multi-source discovery into registered candidates."""

    def __init__(
        self,
        registry: SignalRegistry | None = None,
        *,
        owner: str = "research",
        universe: str = "default",
        frequency: str = "1d",
        auto_register: bool = True,
    ) -> None:
        # Use identity check: empty SignalRegistry is falsy via __len__.
        self.registry = registry if registry is not None else get_default_registry()
        self.owner = owner
        self.universe = universe
        self.frequency = frequency
        self.auto_register = auto_register

    def from_time_series(
        self,
        returns: np.ndarray,
        *,
        volume: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        **kwargs: Any,
    ) -> DiscoveryResult:
        signals = ts_mod.build_time_series_candidates(
            returns,
            volume=volume,
            prices=prices,
            owner=self.owner,
            universe=self.universe,
            frequency=self.frequency,
            **kwargs,
        )
        return self._finalize(signals, notes=["time_series templates"])

    def from_statistical_screen(
        self,
        features: dict[str, np.ndarray],
        target: np.ndarray,
        *,
        min_abs_ic: float = 0.02,
        economic_hypothesis: str | None = None,
    ) -> DiscoveryResult:
        screens = stat_mod.screen_features(
            features, target, min_abs_ic=min_abs_ic, owner=self.owner
        )
        hyp = economic_hypothesis or (
            "Screened association with forward returns is a statistical lead only; "
            "an economic mechanism must be articulated before any promotion. "
            "Statistical significance alone ≠ alpha."
        )
        signals = stat_mod.candidates_to_signals(screens, features, economic_hypothesis=hyp)
        result = self._finalize(signals, notes=["statistical screen — NOT alpha"])
        result.statistical_screens = screens
        return result

    def from_formulas(
        self,
        series: dict[str, np.ndarray],
        formulas: list[tuple[str, list[tuple[str, dict]], str]],
    ) -> DiscoveryResult:
        """Build candidates from symbolic op stacks.

        Each formula tuple: ``(name, ops, economic_hypothesis)``.
        """
        signals: list[AlphaSignal] = []
        for name, ops, hyp in formulas:
            values = sym_mod.evaluate_expression(ops, series)
            definition = SignalDefinition(
                name=name,
                version="0.1.0",
                formula=str(ops),
                features=tuple(series.keys()),
                lookback=1,
                horizon=1,
                universe=self.universe,
                frequency=self.frequency,
                direction="long_short",
                expected_relationship="unknown",
                economic_hypothesis=hyp,
                owner=self.owner,
                signal_type="symbolic",
                tags=("symbolic", "candidate"),
            )
            signals.append(
                AlphaSignal(
                    values=values,
                    name=name,
                    definition_id=definition.definition_id,
                    metadata={
                        "definition": definition.to_dict(),
                        "claims_profitability": False,
                    },
                )
            )
        return self._finalize(signals, notes=["symbolic formulas"])

    def from_forecasts(
        self,
        forecast: np.ndarray,
        *,
        name: str = "forecast_signal",
        economic_hypothesis: str,
        lookback: int = 1,
        horizon: int = 1,
    ) -> DiscoveryResult:
        """Wrap a forecast series as a candidate signal (not auto-approved)."""
        definition = SignalDefinition(
            name=name,
            version="0.1.0",
            formula="forecast_series",
            features=("forecast",),
            lookback=lookback,
            horizon=horizon,
            universe=self.universe,
            frequency=self.frequency,
            direction="long_short",
            expected_relationship="positive",
            economic_hypothesis=economic_hypothesis,
            owner=self.owner,
            signal_type="custom",
            tags=("forecast", "candidate"),
        )
        signal = AlphaSignal(
            values=np.asarray(forecast, dtype=np.float64),
            name=name,
            definition_id=definition.definition_id,
            metadata={
                "definition": definition.to_dict(),
                "claims_profitability": False,
            },
        )
        return self._finalize([signal], notes=["forecast wrapper — candidate only"])

    def from_cross_section(
        self,
        feature_panel: np.ndarray,
        *,
        asset_index: int = 0,
        method: str = "rank",
    ) -> DiscoveryResult:
        if method == "rank":
            sig = cs_mod.cross_sectional_rank_signal(
                feature_panel,
                asset_index=asset_index,
                owner=self.owner,
                universe=self.universe,
                frequency=self.frequency,
            )
        elif method == "zscore":
            sig = cs_mod.cross_sectional_zscore_signal(
                feature_panel,
                asset_index=asset_index,
                owner=self.owner,
                universe=self.universe,
                frequency=self.frequency,
            )
        else:
            raise ValueError(f"Unknown cross-section method: {method}")
        return self._finalize([sig], notes=[f"cross_sectional:{method}"])

    def from_events(
        self,
        event_mask: np.ndarray,
        *,
        returns: np.ndarray | None = None,
        decay: float = 0.5,
        horizon: int = 5,
    ) -> DiscoveryResult:
        signals = [
            event_mod.event_impulse_signal(
                event_mask,
                decay=decay,
                horizon=horizon,
                owner=self.owner,
                universe=self.universe,
                frequency=self.frequency,
            )
        ]
        if returns is not None:
            signals.append(event_mod.earnings_drift_proxy(returns, event_mask, owner=self.owner))
        return self._finalize(signals, notes=["event templates"])

    def from_alternative(
        self,
        alt_series: np.ndarray,
        *,
        publication_lag: int = 1,
        lookback: int = 60,
    ) -> DiscoveryResult:
        signals = [
            alt_mod.alternative_zscore_signal(
                alt_series,
                lookback=lookback,
                publication_lag=publication_lag,
                owner=self.owner,
                universe=self.universe,
                frequency=self.frequency,
            ),
            alt_mod.alternative_change_signal(
                alt_series,
                publication_lag=publication_lag,
                owner=self.owner,
                universe=self.universe,
                frequency=self.frequency,
            ),
        ]
        return self._finalize(signals, notes=["alternative templates"])

    def discover_all(
        self,
        *,
        returns: np.ndarray | None = None,
        features: dict[str, np.ndarray] | None = None,
        target: np.ndarray | None = None,
        volume: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        alt_series: np.ndarray | None = None,
        event_mask: np.ndarray | None = None,
        forecast: np.ndarray | None = None,
        forecast_hypothesis: str | None = None,
    ) -> DiscoveryResult:
        """Run available discovery paths and merge candidates."""
        merged = DiscoveryResult(
            notes=[
                "Merged discovery — all outputs are CANDIDATES, NOT alpha.",
                "Statistical significance alone ≠ alpha.",
                "Historical Sharpe alone cannot approve.",
            ]
        )
        parts: list[DiscoveryResult] = []
        if returns is not None:
            parts.append(self.from_time_series(returns, volume=volume, prices=prices))
        if features is not None and target is not None:
            parts.append(self.from_statistical_screen(features, target))
        if alt_series is not None:
            parts.append(self.from_alternative(alt_series))
        if event_mask is not None:
            parts.append(self.from_events(event_mask, returns=returns))
        if forecast is not None:
            hyp = forecast_hypothesis or (
                "Model forecast encodes predicted conditional expected return from "
                "an estimated mapping; remains a candidate pending economic and "
                "validation review."
            )
            parts.append(self.from_forecasts(forecast, economic_hypothesis=hyp))
        for part in parts:
            merged.signals.extend(part.signals)
            merged.definitions.extend(part.definitions)
            merged.statistical_screens.extend(part.statistical_screens)
            merged.experiment_ids.extend(part.experiment_ids)
            merged.notes.extend(part.notes)
        return merged

    def _finalize(self, signals: list[AlphaSignal], *, notes: list[str]) -> DiscoveryResult:
        definitions: list[SignalDefinition] = []
        experiment_ids: list[str] = []
        for sig in signals:
            raw_def = (sig.metadata or {}).get("definition")
            if isinstance(raw_def, dict):
                definition = SignalDefinition.from_dict(raw_def)
            else:
                definition = SignalDefinition(
                    name=sig.name or "unnamed",
                    version="0.1.0",
                    formula="unknown",
                    features=(),
                    lookback=1,
                    horizon=1,
                    universe=self.universe,
                    frequency=self.frequency,
                    direction="long_short",
                    expected_relationship="unknown",
                    economic_hypothesis=(
                        "Hypothesis pending articulation. "
                        "Statistical significance alone ≠ alpha."
                    ),
                    owner=self.owner,
                    signal_type="custom",
                )
            definitions.append(definition)
            if self.auto_register:
                rec = self.registry.register(
                    definition,
                    signal=sig,
                    status=SignalStatus.CANDIDATE,
                    reason="discovery candidate registration",
                )
                experiment_ids.append(rec.experiment_id)
        return DiscoveryResult(
            signals=signals,
            definitions=definitions,
            experiment_ids=experiment_ids,
            notes=list(notes)
            + [
                "Candidates only — not alpha.",
                "Point-in-time computation required (no future leakage).",
            ],
        )


def generate_candidates(
    returns: np.ndarray,
    *,
    registry: SignalRegistry | None = None,
    **kwargs: Any,
) -> DiscoveryResult:
    """Convenience entrypoint for time-series candidate generation."""
    gen = CandidateGenerator(registry=registry, auto_register=kwargs.pop("auto_register", True))
    return gen.from_time_series(returns, **kwargs)
