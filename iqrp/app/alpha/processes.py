"""Synthetic alpha processes with known ground-truth relationships.

Used for Phase 11 validation and false-discovery-control smoke tests.
These series are research fixtures — not live alpha.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

ScenarioName = Literal[
    "genuine_momentum",
    "random_noise",
    "regime_specific",
    "decaying_signal",
]


def _returns(n: int, rng: np.random.Generator, *, vol: float = 0.01) -> np.ndarray:
    return rng.normal(0.0, vol, size=int(n))


def genuine_momentum(
    n: int = 400,
    *,
    seed: int | None = 0,
    lookback: int = 20,
    signal_strength: float = 0.35,
    noise: float = 0.5,
) -> dict[str, Any]:
    """Lagged momentum signal correlated with forward returns (true alpha DGP)."""
    rng = np.random.default_rng(seed)
    n = int(n)
    innov = rng.normal(0.0, 0.01, size=n)
    # Latent factor that drives future returns; signal observes lagged factor
    factor = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        factor[t] = 0.9 * factor[t - 1] + innov[t]
    returns = factor + rng.normal(0.0, 0.008, size=n)
    # Point-in-time signal: rolling sum of past returns only
    signal = np.full(n, np.nan, dtype=np.float64)
    for t in range(lookback, n):
        past = returns[t - lookback : t]
        signal[t] = float(np.sum(past))
    # Inject known predictive link: mix signal with next-period factor (for truth meta)
    # Keep PIT: use factor[t] (known at t) which correlates with returns[t+1]
    predictive = signal_strength * factor + noise * rng.normal(0.0, 1.0, size=n)
    # Blend with momentum so evaluate/validate see IC > 0
    m = np.isfinite(signal)
    signal = np.where(m, 0.6 * signal + 0.4 * predictive, predictive)
    return {
        "name": "genuine_momentum",
        "signal": signal,
        "returns": returns,
        "truth": {
            "is_alpha": True,
            "expected_pass_validation": True,
            "relationship": "positive",
            "lookback": lookback,
            "signal_strength": signal_strength,
            "notes": "Lagged factor / momentum correlated with forward returns.",
        },
    }


def random_noise(n: int = 400, *, seed: int | None = 0) -> dict[str, Any]:
    """IID noise signal independent of returns — should fail validation."""
    rng = np.random.default_rng(seed)
    returns = _returns(n, rng)
    signal = rng.normal(0.0, 1.0, size=int(n))
    return {
        "name": "random_noise",
        "signal": signal,
        "returns": returns,
        "truth": {
            "is_alpha": False,
            "expected_pass_validation": False,
            "relationship": "none",
            "notes": "Pure noise; false-discovery control should reject.",
        },
    }


def regime_specific(
    n: int = 400,
    *,
    seed: int | None = 0,
    active_regime: str = "risk_on",
) -> dict[str, Any]:
    """Signal predictive only in one regime."""
    rng = np.random.default_rng(seed)
    n = int(n)
    regimes = np.asarray(
        ["risk_on" if i % 2 == 0 else "risk_off" for i in range(n)],
        dtype=object,
    )
    # Alternate blocks for clearer regimes
    block = max(n // 8, 10)
    for i in range(0, n, block):
        label = "risk_on" if (i // block) % 2 == 0 else "risk_off"
        regimes[i : i + block] = label

    latent = rng.normal(0.0, 1.0, size=n)
    signal = latent + rng.normal(0.0, 0.2, size=n)
    returns = rng.normal(0.0, 0.01, size=n)
    active = regimes == active_regime
    # In active regime, forward return correlates with signal
    noise = rng.normal(0.0, 0.008, size=n)
    returns = np.where(active, 0.015 * signal + noise, noise)
    return {
        "name": "regime_specific",
        "signal": signal,
        "returns": returns,
        "regimes": regimes,
        "truth": {
            "is_alpha": True,
            "expected_pass_validation": True,
            "relationship": "regime_conditional",
            "active_regime": active_regime,
            "notes": f"Predictive only in regime={active_regime}.",
        },
    }


def decaying_signal(
    n: int = 400,
    *,
    seed: int | None = 0,
    half_life: float = 5.0,
) -> dict[str, Any]:
    """Signal whose IC decays with horizon (known half-life)."""
    rng = np.random.default_rng(seed)
    n = int(n)
    innov = rng.normal(0.0, 1.0, size=n)
    signal = innov.copy()
    # Build returns as exponentially decaying response to lagged signal
    lam = np.log(2.0) / max(float(half_life), 1e-6)
    returns = np.zeros(n, dtype=np.float64)
    for h in range(1, min(40, n)):
        w = np.exp(-lam * h)
        returns[h:] += w * 0.01 * signal[:-h]
    returns += rng.normal(0.0, 0.005, size=n)
    return {
        "name": "decaying_signal",
        "signal": signal,
        "returns": returns,
        "truth": {
            "is_alpha": True,
            "expected_pass_validation": True,
            "relationship": "decaying_positive",
            "half_life": float(half_life),
            "notes": "IC decays with horizon near configured half_life.",
        },
    }


_SCENARIOS = {
    "genuine_momentum": genuine_momentum,
    "random_noise": random_noise,
    "regime_specific": regime_specific,
    "decaying_signal": decaying_signal,
}


def simulate_alpha_scenario(
    name: ScenarioName | str,
    n: int = 400,
    *,
    seed: int | None = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dispatch synthetic scenario by name; always includes ``truth`` metadata."""
    key = str(name).strip().lower()
    if key not in _SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. Available: {sorted(_SCENARIOS)}"
        )
    out = _SCENARIOS[key](n, seed=seed, **kwargs)
    out.setdefault("n", int(n))
    out.setdefault("seed", seed)
    return out


def available_scenarios() -> list[str]:
    return sorted(_SCENARIOS)
