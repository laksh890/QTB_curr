"""Dynamic discovery of regime models for the ensemble (no hard-coded imports)."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.regimes.base.regime_model import RegimeModel
from iqrp.app.regimes.base.registry import get_registry
from iqrp.app.regimes.ensemble.config import EnsembleSettings

logger = logging.getLogger(__name__)

# Names that must never be nested into the ensemble as members
_EXCLUDED = frozenset({"ensemble", "ensemble_regime"})


# Alias table: member state name → canonical regime name
_NAME_ALIASES: dict[str, str] = {
    "bull": "bull",
    "bullish": "bull",
    "up": "bull",
    "bear": "bear",
    "bearish": "bear",
    "down": "bear",
    "sideways": "sideways",
    "neutral": "sideways",
    "range": "sideways",
    "high_volatility": "high_volatility",
    "high_vol": "high_volatility",
    "volatile": "high_volatility",
    "low_volatility": "low_volatility",
    "low_vol": "low_volatility",
    "calm": "low_volatility",
    "liquidity_stress": "liquidity_stress",
    "stress": "liquidity_stress",
    "illiquid": "liquidity_stress",
}


@dataclass
class EnsembleMember:
    """A fitted (or fit-ready) regime model with mapping into canonical space."""

    name: str
    model: RegimeModel
    weight: float = 1.0
    state_map: np.ndarray | None = None  # (K_member, K_canonical) soft map
    metadata: dict[str, Any] = field(default_factory=dict)

    def map_proba(self, proba: np.ndarray, n_canonical: int) -> np.ndarray:
        """Map member probabilities ``(T, K_m)`` into canonical ``(T, K_c)``."""
        p = np.asarray(proba, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        if self.state_map is not None:
            mapped = p @ self.state_map
        else:
            mapped = _default_map(p, n_canonical)
        row = mapped.sum(axis=1, keepdims=True)
        row = np.clip(row, 1e-300, None)
        return mapped / row


def discover_modules(modules: tuple[str, ...] | list[str]) -> list[str]:
    """Import modules for side-effect registration; return successfully loaded names."""
    loaded: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except Exception as exc:  # pragma: no cover - optional deps
            logger.warning("Ensemble discovery skipped module %s: %s", mod, exc)
    return loaded


def list_available_members(*, exclude: frozenset[str] = _EXCLUDED) -> list[str]:
    names = get_registry().list_names()
    return [n for n in names if n not in exclude]


def build_state_map(
    member_names: tuple[str, ...] | list[str],
    canonical_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Build ``(K_m, K_c)`` soft assignment from name aliases / index fallback."""
    canon = [str(c).lower() for c in canonical_names]
    k_c = len(canon)
    k_m = len(member_names)
    m = np.zeros((k_m, k_c), dtype=np.float64)
    for i, name in enumerate(member_names):
        key = _NAME_ALIASES.get(str(name).lower(), str(name).lower())
        if key in canon:
            m[i, canon.index(key)] = 1.0
        elif i < k_c:
            m[i, i] = 1.0
        else:
            # spread residual mass uniformly
            m[i, :] = 1.0 / k_c
    # rows with no assignment (defensive; all rows assigned above)
    for i in range(k_m):  # pragma: no cover
        if m[i].sum() <= 0:
            m[i, :] = 1.0 / k_c
    return m


def _default_map(proba: np.ndarray, n_canonical: int) -> np.ndarray:
    t, k = proba.shape
    out = np.zeros((t, n_canonical), dtype=np.float64)
    if k == n_canonical:
        return proba.copy()
    if k == 2 and n_canonical >= 3:
        # bearish/bullish → bear, bull, sideways from confidence
        conf = np.abs(proba[:, 1] - proba[:, 0])
        sideways = np.clip(1.0 - conf, 0.0, 1.0)
        remain = 1.0 - sideways
        # assume col0 bearish, col1 bullish
        out[:, 1] = remain * proba[:, 0]  # bear
        out[:, 0] = remain * proba[:, 1]  # bull
        out[:, 2] = sideways
        return out
    n = min(k, n_canonical)
    out[:, :n] = proba[:, :n]
    if k > n_canonical:
        # fold extras into last canonical bin
        out[:, -1] += proba[:, n_canonical:].sum(axis=1)
    return out


class EnsembleRegistry:
    """Construct ensemble members from the global regime registry."""

    def __init__(self, settings: EnsembleSettings | None = None) -> None:
        self.settings = settings or EnsembleSettings.default()

    def discover(self) -> list[str]:
        discover_modules(self.settings.discovery_modules)
        available = list_available_members()
        if self.settings.member_names:
            wanted = [n for n in self.settings.member_names if n in available]
            return wanted
        return available

    def create_members(self, **create_kwargs: Any) -> list[EnsembleMember]:
        names = self.discover()
        if len(names) < self.settings.training.min_members:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Need at least {self.settings.training.min_members} ensemble members, "
                f"found {names}",
                code="ENS_NO_MEMBERS",
            )
        members: list[EnsembleMember] = []
        canon = self.settings.state_names
        for name in names:
            try:
                model = get_registry().create(name, **create_kwargs)
            except TypeError:
                model = get_registry().create(name)
            member_names = model.meta.state_names or tuple(
                f"state_{i}" for i in range(model.meta.n_states)
            )
            smap = build_state_map(member_names, canon)
            members.append(
                EnsembleMember(name=name, model=model, weight=1.0, state_map=smap)
            )
        # equal weights
        w = 1.0 / max(len(members), 1)
        for m in members:
            m.weight = w
        return members
