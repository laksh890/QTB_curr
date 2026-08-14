"""Dynamic risk budgeting from confidence / regime / vol / liquidity / drawdown / risk_state / capacity.

Confidence cannot expand scale beyond 1.0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.capital.capacity import estimate_capacity
from iqrp.app.risk.capital.config import CapitalSettings
from iqrp.app.risk.capital.correlation import correlation_crowding_scales, strategy_correlation
from iqrp.app.risk.capital.drawdown import drawdown_scales
from iqrp.app.risk.capital.volatility import volatility_budgets

_DEFAULT_RISK_STATE = {
    "NORMAL": 1.0,
    "CAUTION": 0.8,
    "REDUCED_RISK": 0.5,
    "CAPITAL_PRESERVATION": 0.25,
    "TRADING_HALT": 0.0,
}


def dynamic_risk_scales(
    names: list[str],
    *,
    settings: CapitalSettings | None = None,
    cov: np.ndarray | None = None,
    returns: np.ndarray | None = None,
    vols: np.ndarray | list[float] | None = None,
    adv: np.ndarray | list[float] | None = None,
    spreads: np.ndarray | list[float] | None = None,
    drawdowns: np.ndarray | list[float] | None = None,
    forecast_confidence: np.ndarray | list[float] | None = None,
    model_agreement: np.ndarray | list[float] | None = None,
    expected_opportunity: np.ndarray | list[float] | None = None,
    regime: str = "normal",
    risk_state: str = "NORMAL",
    capital: float = 1.0,
    base_weights: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Compute per-name dynamic scales and combined weight multipliers."""
    cfg = settings or CapitalSettings.default()
    n = len(names)
    if n == 0:
        return {"name": "dynamic_risk_scales", "scales": {}, "weights": {}}

    # Portfolio-level risk state (hard ceiling)
    rs_key = str(risk_state).upper()
    rs_scales = dict(_DEFAULT_RISK_STATE)
    rs_scales.update({str(k).upper(): float(v) for k, v in cfg.risk_state_scales.items()})
    portfolio_scale = float(np.clip(rs_scales.get(rs_key, 0.5), 0.0, 1.0))

    # Regime scale (cannot expand beyond 1)
    reg_key = str(regime).lower()
    regime_scale = float(np.clip(cfg.regime_scales.get(reg_key, 0.5), 0.0, 1.0))

    # Drawdown
    dd = drawdown_scales(
        names,
        returns=returns,
        drawdowns=drawdowns,
        caution=cfg.drawdown.caution,
        reduced_risk=cfg.drawdown.reduced_risk,
        capital_preservation=cfg.drawdown.capital_preservation,
        trading_halt=cfg.drawdown.trading_halt,
        state_scales=cfg.risk_state_scales,
    )
    dd_scales = dd["scales"]

    # Capacity / liquidity
    cap = estimate_capacity(
        names,
        capital=capital,
        weights=base_weights,
        adv=adv,
        spreads=spreads,
        vols=vols,
        max_participation=cfg.max_participation,
        impact_coeff=cfg.impact_coeff,
        ttl_days=cfg.capacity_ttl_days,
        missing_capacity_scale=cfg.missing_capacity_scale,
        missing_liquidity_scale=cfg.missing_liquidity_scale,
        default_adv=cfg.default_adv,
        default_spread=cfg.default_spread,
    )
    cap_scales = cap["scales"]

    # Correlation crowding
    corr_m = None
    if returns is not None:
        corr_m = np.asarray(strategy_correlation(returns)["matrix"], dtype=np.float64)
    elif cov is not None:
        c = np.asarray(cov, dtype=np.float64)
        vol = np.sqrt(np.maximum(np.diag(c), 1e-18))
        denom = np.outer(vol, vol)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_m = np.where(denom > 0, c / denom, 0.0)
        np.fill_diagonal(corr_m, 1.0)
    else:
        corr_m = np.eye(n)
    corr_scales = correlation_crowding_scales(
        corr_m,
        threshold=cfg.correlation_crowding_threshold,
        floor=cfg.correlation_scale_floor,
        names=names,
    )

    # Vol relative scale (high vol → lower budget share already in inv-vol; here damp extremes)
    vol_info = volatility_budgets(
        names,
        vols=vols,
        returns=returns,
        cov=cov,
        target_volatility=cfg.target_volatility,
        vol_floor=cfg.vol_floor,
    )
    vol_vals = np.asarray([vol_info["vols"][nm] for nm in names], dtype=np.float64)
    med = float(np.median(vol_vals)) if n else 0.01
    med = max(med, cfg.vol_floor)
    vol_scales = {
        names[i]: float(np.clip(med / max(vol_vals[i], cfg.vol_floor), 0.25, 1.0)) for i in range(n)
    }

    # Confidence / agreement — CAP at 1.0 (cannot expand)
    conf = _optional_unit(forecast_confidence, n, default=1.0)
    agree = _optional_unit(model_agreement, n, default=1.0)
    # Opportunity tilt only if explicitly provided (NOT historical mean)
    opp = None
    if expected_opportunity is not None:
        opp = np.asarray(expected_opportunity, dtype=np.float64).ravel()
        if opp.size != n:
            opp = None
        else:
            opp = np.maximum(opp, 0.0)
            if float(np.sum(opp)) > 0:
                opp = opp / float(np.sum(opp))
            else:
                opp = np.full(n, 1.0 / n)

    scales: dict[str, float] = {}
    for i, nm in enumerate(names):
        # Confidence mixes with agreement but never exceeds 1
        conf_i = float(np.clip(min(conf[i], agree[i]), 0.0, 1.0))
        # Map confidence to [0.25, 1.0] — floor only, no expansion above 1
        conf_scale = 0.25 + 0.75 * conf_i
        s = (
            portfolio_scale
            * regime_scale
            * float(dd_scales.get(nm, 1.0))
            * float(cap_scales.get(nm, 1.0))
            * float(corr_scales.get(nm, 1.0))
            * float(vol_scales.get(nm, 1.0))
            * conf_scale
        )
        scales[nm] = float(np.clip(s, 0.0, 1.0))

    # Base weights: inv-vol or equal, then opportunity tilt, then dynamic scales
    base = np.asarray(
        [float(vol_info["weights"].get(nm, 1.0 / n)) for nm in names], dtype=np.float64
    )
    if base_weights is not None:
        bw = np.asarray(base_weights, dtype=np.float64).ravel()
        if bw.size == n:
            base = np.maximum(bw, 0.0)
            if float(np.sum(base)) > 0:
                base = base / float(np.sum(base))
    if opp is not None:
        base = base * opp
        if float(np.sum(base)) > 0:
            base = base / float(np.sum(base))

    scaled = np.asarray([base[i] * scales[names[i]] for i in range(n)], dtype=np.float64)
    tot = float(np.sum(scaled))
    if tot > 1e-12:
        scaled = scaled / tot
    else:
        # Trading halt / total collapse
        scaled = np.zeros(n, dtype=np.float64)

    return {
        "name": "dynamic_risk_scales",
        "scales": scales,
        "weights": {names[i]: float(scaled[i]) for i in range(n)},
        "weight_vector": scaled.tolist(),
        "portfolio_scale": portfolio_scale,
        "regime_scale": regime_scale,
        "drawdown_scales": dd_scales,
        "capacity_scales": cap_scales,
        "correlation_scales": corr_scales,
        "volatility_scales": vol_scales,
        "confidence": {names[i]: float(conf[i]) for i in range(n)},
        "model_agreement": {names[i]: float(agree[i]) for i in range(n)},
        "opportunity_applied": opp is not None,
        "risk_state": rs_key,
        "regime": reg_key,
    }


def _optional_unit(
    values: np.ndarray | list[float] | None,
    n: int,
    *,
    default: float,
) -> np.ndarray:
    if values is None:
        return np.full(n, float(default), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size != n:
        return np.full(n, float(default), dtype=np.float64)
    return np.clip(arr, 0.0, 1.0)
