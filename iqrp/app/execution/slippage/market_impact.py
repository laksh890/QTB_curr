"""Square-root market impact aligned with the simulation SlippageModel."""

from __future__ import annotations

from typing import Any

import numpy as np


def _load_simulation_slippage_model() -> type | None:
    """Import-only alignment with ``iqrp.app.simulation.liquidity.slippage.SlippageModel``.

    Loaded lazily so execution TCA remains usable even when the simulation package
    cannot be imported (e.g. older Python / incomplete env).
    """
    try:
        from iqrp.app.simulation.liquidity.slippage import SlippageModel as SimulationSlippageModel

        return SimulationSlippageModel
    except Exception:  # noqa: BLE001 — optional dependency surface
        return None


# Public name for callers / type checkers; may be None if simulation unavailable.
SimulationSlippageModel = _load_simulation_slippage_model()


def market_impact(
    *,
    side: str,
    quantity: float,
    mid: float,
    adv: float,
    volatility: float = 0.02,
    spread: float = 0.0,
    impact_coeff: float = 0.1,
    permanent_ratio: float = 0.5,
    use_simulation_model: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Temporary / permanent market impact in price units and bps.

    When ``use_simulation_model`` is True and the simulation engine is importable,
    delegates to ``iqrp.app.simulation.liquidity.slippage.SlippageModel``.
    """
    qty = abs(float(quantity))
    mid_f = max(float(mid), 1e-12)
    adv_f = max(float(adv), 1e-12)
    vol = max(float(volatility), 0.0)
    participation = qty / adv_f

    sim_cls = SimulationSlippageModel if SimulationSlippageModel is not None else _load_simulation_slippage_model()
    if use_simulation_model and sim_cls is not None:
        sim = sim_cls(impact=impact_coeff, rng=rng)
        out = sim.execution_price(
            mid_f,
            side,
            qty,
            adv=adv_f,
            volatility=vol,
            spread=float(spread),
        )
        slip = abs(float(out["price"]) - mid_f)
        return {
            "name": "market_impact",
            "temporary_impact": float(out["temporary_impact"]),
            "permanent_impact": float(out["permanent_impact"]),
            "slippage": float(slip),
            "slippage_bps": float(slip / mid_f * 1e4),
            "participation": float(out["participation"]),
            "price": float(out["price"]),
            "source": "simulation.SlippageModel",
        }

    temp = float(impact_coeff) * vol * mid_f * float(np.sqrt(participation))
    perm = float(permanent_ratio) * temp
    total = temp
    return {
        "name": "market_impact",
        "temporary_impact": float(temp),
        "permanent_impact": float(perm),
        "slippage": float(total),
        "slippage_bps": float(total / mid_f * 1e4),
        "participation": float(participation),
        "impact_coeff": float(impact_coeff),
        "source": "analytic",
    }


def path_impact(
    mids: Any,
    volumes: Any,
    trade_sizes: Any,
    volatility: Any,
    *,
    impact_coeff: float = 0.1,
) -> np.ndarray:
    """Vectorized temporary impact series (price units), simulation-aligned."""
    sim_cls = SimulationSlippageModel if SimulationSlippageModel is not None else _load_simulation_slippage_model()
    if sim_cls is not None:
        return sim_cls(impact=impact_coeff).path_impact(mids, volumes, trade_sizes, volatility)

    mids_a = np.asarray(mids, dtype=np.float64)
    vols = np.asarray(volumes, dtype=np.float64)
    sizes = np.asarray(trade_sizes, dtype=np.float64)
    vol = np.asarray(volatility, dtype=np.float64)
    n = min(len(mids_a), len(vols), len(sizes), len(vol))
    part = np.abs(sizes[:n]) / np.maximum(vols[:n], 1e-8)
    return float(impact_coeff) * vol[:n] * mids_a[:n] * np.sqrt(part)


__all__ = ["SimulationSlippageModel", "market_impact", "path_impact"]
