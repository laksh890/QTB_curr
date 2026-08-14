"""Calibration diagnostics for VaR, ES, volatility, liquidity, and drawdown."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import as_returns
from iqrp.app.risk.ensemble.config import EnsembleSettings


def _finite_pair(predicted: Any, realized: Any) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    r = np.asarray(realized, dtype=np.float64).reshape(-1)
    n = min(p.size, r.size)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    p, r = p[-n:], r[-n:]
    mask = np.isfinite(p) & np.isfinite(r)
    return p[mask], r[mask]


def var_calibration(
    predicted_var: Any,
    realized_returns: Any,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """VaR exceedance rate vs nominal alpha.

    Convention: VaR is a positive loss quantile; exceedance when -return > VaR
    (or return < -VaR).
    """
    var, rets = _finite_pair(predicted_var, realized_returns)
    if var.size == 0:
        return {
            "name": "var_calibration",
            "n_obs": 0,
            "exceedance_rate": None,
            "nominal_alpha": float(alpha),
            "exceedance_bias": None,
            "n_exceedances": 0,
            "calibrated": None,
        }
    # Support signed VaR (negative loss) or positive loss magnitude
    loss = -rets
    var_mag = np.abs(var)
    exceed = loss > var_mag
    rate = float(np.mean(exceed))
    bias = rate - float(alpha)
    return {
        "name": "var_calibration",
        "n_obs": int(var.size),
        "exceedance_rate": rate,
        "nominal_alpha": float(alpha),
        "exceedance_bias": float(bias),
        "n_exceedances": int(np.sum(exceed)),
        "mean_predicted_var": float(np.mean(var_mag)),
        "calibrated": bool(abs(bias) <= max(float(alpha) * 0.5, 0.01)),
    }


def es_calibration(
    predicted_es: Any,
    realized_returns: Any,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Expected shortfall calibration: mean loss given VaR exceedance vs predicted ES."""
    es, rets = _finite_pair(predicted_es, realized_returns)
    if es.size == 0:
        return {
            "name": "es_calibration",
            "n_obs": 0,
            "realized_es": None,
            "predicted_es": None,
            "es_error": None,
            "calibrated": None,
        }
    loss = -rets
    es_mag = np.abs(es)
    # Approximate tail set via empirical alpha quantile of loss
    thresh = float(np.quantile(loss, 1.0 - float(alpha))) if loss.size else 0.0
    tail = loss[loss >= thresh]
    realized_es = float(np.mean(tail)) if tail.size else float(np.mean(loss))
    predicted = float(np.mean(es_mag))
    err = realized_es - predicted
    return {
        "name": "es_calibration",
        "n_obs": int(es.size),
        "realized_es": realized_es,
        "predicted_es": predicted,
        "es_error": float(err),
        "relative_error": float(err / max(abs(predicted), 1e-12)),
        "calibrated": bool(abs(err) <= max(0.02, 0.25 * abs(predicted))),
    }


def vol_calibration(predicted_vol: Any, realized_vol: Any) -> dict[str, Any]:
    p, r = _finite_pair(predicted_vol, realized_vol)
    if p.size == 0:
        return {
            "name": "vol_calibration",
            "n_obs": 0,
            "rmse": None,
            "bias": None,
            "calibration_error": None,
            "calibrated": None,
        }
    err = p - r
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    rel = float(np.mean(np.abs(err) / np.maximum(np.abs(r), 1e-12)))
    return {
        "name": "vol_calibration",
        "n_obs": int(p.size),
        "rmse": rmse,
        "bias": bias,
        "calibration_error": rel,
        "calibrated": bool(rel <= 0.25),
    }


def liquidity_calibration(predicted_liq: Any, observed_liq: Any) -> dict[str, Any]:
    p, o = _finite_pair(predicted_liq, observed_liq)
    if p.size == 0:
        return {
            "name": "liquidity_calibration",
            "n_obs": 0,
            "mae": None,
            "bias": None,
            "calibrated": None,
        }
    err = p - o
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return {
        "name": "liquidity_calibration",
        "n_obs": int(p.size),
        "mae": mae,
        "bias": bias,
        "calibrated": bool(mae <= 0.15),
    }


def drawdown_calibration(predicted_dd: Any, realized_dd: Any) -> dict[str, Any]:
    p, r = _finite_pair(predicted_dd, realized_dd)
    if p.size == 0:
        return {
            "name": "drawdown_calibration",
            "n_obs": 0,
            "mae": None,
            "underestimation_rate": None,
            "calibrated": None,
        }
    err = p - r
    mae = float(np.mean(np.abs(err)))
    under = float(np.mean(p < r))
    return {
        "name": "drawdown_calibration",
        "n_obs": int(p.size),
        "mae": mae,
        "bias": float(np.mean(err)),
        "underestimation_rate": under,
        "calibrated": bool(mae <= 0.05 and under <= 0.6),
    }


def run_calibration(
    *,
    settings: EnsembleSettings,
    predicted_var: Any = None,
    predicted_es: Any = None,
    predicted_vol: Any = None,
    realized_vol: Any = None,
    predicted_liquidity: Any = None,
    observed_liquidity: Any = None,
    predicted_drawdown: Any = None,
    realized_drawdown: Any = None,
    realized_returns: Any = None,
) -> dict[str, Any]:
    """Aggregate calibration statistics used by weighting and diagnostics."""
    alpha_var = float(settings.calibration.var_alpha)
    alpha_es = float(settings.calibration.es_alpha)
    out: dict[str, Any] = {"tolerance_band": float(settings.calibration.tolerance_band)}

    if predicted_var is not None and realized_returns is not None:
        out["var"] = var_calibration(predicted_var, realized_returns, alpha=alpha_var)
        out["var_exceedance_bias"] = out["var"].get("exceedance_bias")
    if predicted_es is not None and realized_returns is not None:
        out["es"] = es_calibration(predicted_es, realized_returns, alpha=alpha_es)
        out["es_error"] = out["es"].get("es_error")
    if predicted_vol is not None and realized_vol is not None:
        out["vol"] = vol_calibration(predicted_vol, realized_vol)
        out["vol_calibration_error"] = out["vol"].get("calibration_error")
    if predicted_liquidity is not None and observed_liquidity is not None:
        out["liquidity"] = liquidity_calibration(predicted_liquidity, observed_liquidity)
    if predicted_drawdown is not None and realized_drawdown is not None:
        out["drawdown"] = drawdown_calibration(predicted_drawdown, realized_drawdown)
    elif realized_returns is not None and predicted_drawdown is not None:
        # Derive realized drawdown path length-matched if series provided
        from iqrp.app.risk.tail.drawdown import drawdown_series

        dd = drawdown_series(as_returns(realized_returns))
        out["drawdown"] = drawdown_calibration(predicted_drawdown, dd)

    flags = []
    for key in ("var", "es", "vol", "liquidity", "drawdown"):
        block = out.get(key)
        if isinstance(block, dict) and block.get("calibrated") is False:
            flags.append(key)
    out["miscalibrated"] = flags
    out["all_calibrated"] = len(flags) == 0 and any(
        k in out for k in ("var", "es", "vol", "liquidity", "drawdown")
    )
    return out


class CalibrationEngine:
    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        return run_calibration(settings=self.settings, **kwargs)
