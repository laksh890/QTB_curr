"""Simulation scenario specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from iqrp.app.simulation.config import SimulationSettings


@dataclass
class Scenario:
    """Fully parameterized market simulation scenario."""

    name: str = "default"
    model: str = "gbm"
    n_steps: int = 1000
    n_assets: int = 1
    dt: float = 0.004
    initial_price: float = 100.0
    asset_class: Literal["stock", "crypto", "forex", "commodity", "index"] = "crypto"
    random_seed: int = 42
    symbols: tuple[str, ...] = ("SYNTH",)
    correlation_matrix: np.ndarray | None = None
    transition_matrix: np.ndarray | None = None
    regime_enabled: bool = True
    events_enabled: bool = True
    noise_distribution: str = "gaussian"
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        settings: SimulationSettings | None = None,
        *,
        name: str = "default",
        model: str | None = None,
        **overrides: Any,
    ) -> Scenario:
        settings = settings or SimulationSettings.default()
        n_assets = int(overrides.pop("n_assets", settings.n_assets))
        default_symbols = tuple(f"{settings.market.symbol}_{i}" for i in range(n_assets))
        symbols = tuple(overrides.pop("symbols", default_symbols))
        if n_assets == 1 and len(symbols) == 1 and symbols[0].endswith("_0"):
            symbols = (settings.market.symbol,)
        rho = float(settings.correlation.rho)
        corr = _default_correlation(n_assets, rho)
        k = settings.regimes.n_states
        p = float(settings.regimes.persistence)
        off = (1.0 - p) / max(k - 1, 1)
        tm = np.full((k, k), off, dtype=np.float64)
        np.fill_diagonal(tm, p)
        return cls(
            name=name,
            model=model or settings.default_model,
            n_steps=int(overrides.pop("n_steps", settings.n_steps)),
            n_assets=n_assets,
            dt=float(overrides.pop("dt", settings.dt)),
            initial_price=float(overrides.pop("initial_price", settings.initial_price)),
            asset_class=overrides.pop("asset_class", settings.asset_class),
            random_seed=int(overrides.pop("random_seed", settings.random_seed)),
            symbols=symbols[:n_assets] if len(symbols) >= n_assets else symbols,
            correlation_matrix=overrides.pop("correlation_matrix", corr),
            transition_matrix=overrides.pop("transition_matrix", tm),
            regime_enabled=bool(overrides.pop("regime_enabled", settings.regimes.enabled)),
            events_enabled=bool(overrides.pop("events_enabled", settings.events.enabled)),
            noise_distribution=str(
                overrides.pop("noise_distribution", settings.noise.distribution)
            ),
            parameters={
                "drift": settings.dynamics.drift,
                "volatility": settings.dynamics.volatility,
                "mean_reversion_speed": settings.dynamics.mean_reversion_speed,
                "mean_reversion_level": settings.dynamics.mean_reversion_level,
                "jump_intensity": settings.dynamics.jump_intensity,
                "jump_mean": settings.dynamics.jump_mean,
                "jump_std": settings.dynamics.jump_std,
                "heston_kappa": settings.dynamics.heston_kappa,
                "heston_theta": settings.dynamics.heston_theta,
                "heston_xi": settings.dynamics.heston_xi,
                "heston_rho": settings.dynamics.heston_rho,
                "vg_theta": settings.dynamics.vg_theta,
                "vg_sigma": settings.dynamics.vg_sigma,
                "vg_nu": settings.dynamics.vg_nu,
                "cir_kappa": settings.dynamics.cir_kappa,
                "cir_theta": settings.dynamics.cir_theta,
                "cir_sigma": settings.dynamics.cir_sigma,
                "regime_drifts": list(settings.regimes.drifts),
                "regime_volatilities": list(settings.regimes.volatilities),
                "regime_names": list(settings.regimes.state_names),
                **dict(overrides.pop("parameters", {})),
            },
            metadata=dict(overrides.pop("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.correlation_matrix is not None:
            d["correlation_matrix"] = np.asarray(self.correlation_matrix).tolist()
        if self.transition_matrix is not None:
            d["transition_matrix"] = np.asarray(self.transition_matrix).tolist()
        return d


def _default_correlation(n: int, rho: float) -> np.ndarray:
    if n <= 1:
        return np.ones((1, 1), dtype=np.float64)
    mat = np.full((n, n), float(rho), dtype=np.float64)
    np.fill_diagonal(mat, 1.0)
    # Ensure PSD via eigenvalue clip
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 1e-8, None)
    mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Rescale diagonal to 1
    d = np.sqrt(np.diag(mat))
    mat = mat / np.outer(d, d)
    return np.asarray(mat, dtype=np.float64)
