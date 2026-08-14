"""Parameter robustness, sensitivity, and ablation testing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

__all__ = [
    "ablation_test",
    "overfitting_risk",
    "parameter_sweep",
    "sensitivity_analysis",
    "stability_regions",
]


MetricFn = Callable[[np.ndarray], float]


def _default_metrics(returns: np.ndarray, *, periods_per_year: float = 252.0) -> dict[str, float]:
    return {
        "total_return": total_return(returns),
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
    }


def parameter_sweep(
    objective: Callable[..., Any],
    param_grid: Mapping[str, Sequence[Any]],
    *,
    periods_per_year: float = 252.0,
    metric_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Sweep a parameter grid through ``objective(**params) -> returns | dict``.

    ``objective`` must return a return series or a dict containing ``returns``.
    """
    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    rows: list[dict[str, Any]] = []

    for combo in product(*values):
        params = dict(zip(keys, combo))
        out = objective(**params)
        if isinstance(out, Mapping):
            r = as_returns(out["returns"])
            extra = {k: v for k, v in out.items() if k != "returns"}
        else:
            r = as_returns(out)
            extra = {}
        metrics = _default_metrics(r, periods_per_year=periods_per_year)
        if metric_keys:
            metrics = {k: metrics[k] for k in metric_keys if k in metrics}
        rows.append({**params, **metrics, **extra})

    if not rows:
        return {"name": "parameter_sweep", "results": [], "surface": {}}

    # Performance surface for first two numeric params if available
    surface: dict[str, Any] = {}
    numeric_keys = [
        k
        for k in keys
        if all(isinstance(rows[i][k], (int, float, np.floating)) for i in range(len(rows)))
    ]
    if len(numeric_keys) >= 1:
        xk = numeric_keys[0]
        surface["x"] = xk
        surface["x_values"] = [row[xk] for row in rows]
        surface["sharpe"] = [row.get("sharpe", 0.0) for row in rows]
    if len(numeric_keys) >= 2:
        yk = numeric_keys[1]
        surface["y"] = yk
        surface["y_values"] = [row[yk] for row in rows]

    sharpes = np.asarray([row.get("sharpe", 0.0) for row in rows], dtype=np.float64)
    best_idx = int(np.nanargmax(sharpes)) if sharpes.size else 0
    return {
        "name": "parameter_sweep",
        "results": rows,
        "surface": surface,
        "best": rows[best_idx] if rows else None,
        "n_combinations": len(rows),
    }


def sensitivity_analysis(
    objective: Callable[..., Any],
    base_params: Mapping[str, Any],
    *,
    scales: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.2),
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """One-at-a-time sensitivity around ``base_params`` for numeric knobs."""
    base_out = objective(**dict(base_params))
    base_r = as_returns(base_out["returns"] if isinstance(base_out, Mapping) else base_out)
    base_metrics = _default_metrics(base_r, periods_per_year=periods_per_year)

    sensitivities: dict[str, Any] = {}
    for key, val in base_params.items():
        if not isinstance(val, (int, float, np.floating)):
            continue
        curve = []
        for s in scales:
            params = dict(base_params)
            params[key] = type(val)(float(val) * float(s)) if not isinstance(val, bool) else val
            out = objective(**params)
            r = as_returns(out["returns"] if isinstance(out, Mapping) else out)
            m = _default_metrics(r, periods_per_year=periods_per_year)
            curve.append({"scale": float(s), "value": params[key], **m})
        sharpes = [c["sharpe"] for c in curve]
        sensitivities[key] = {
            "curve": curve,
            "sharpe_range": float(np.nanmax(sharpes) - np.nanmin(sharpes)) if sharpes else 0.0,
        }

    return {
        "name": "sensitivity",
        "base": dict(base_params),
        "base_metrics": base_metrics,
        "sensitivities": sensitivities,
    }


def ablation_test(
    objective: Callable[..., Any],
    *,
    components: Mapping[str, bool],
    base_params: Mapping[str, Any] | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Ablate components (features/signals/models/risk/execution/costs).

    ``objective`` receives ``base_params`` plus boolean flags from ``components``.
    Each key is toggled off individually (and all-on baseline is included).
    """
    base = dict(base_params or {})
    results: list[dict[str, Any]] = []

    # Full configuration
    full_flags = dict.fromkeys(components, True)
    full_flags.update({k: bool(v) for k, v in components.items()})
    out = objective(**{**base, **full_flags})
    r = as_returns(out["returns"] if isinstance(out, Mapping) else out)
    results.append(
        {"ablation": "none", **full_flags, **_default_metrics(r, periods_per_year=periods_per_year)}
    )

    for key in components:
        flags = dict(full_flags)
        flags[key] = False
        out = objective(**{**base, **flags})
        r = as_returns(out["returns"] if isinstance(out, Mapping) else out)
        metrics = _default_metrics(r, periods_per_year=periods_per_year)
        delta_sharpe = float(metrics["sharpe"] - results[0]["sharpe"])
        results.append(
            {
                "ablation": key,
                **flags,
                **metrics,
                "delta_sharpe": delta_sharpe,
            }
        )

    return {"name": "ablation", "results": results}


def stability_regions(
    sweep_result: Mapping[str, Any],
    *,
    min_sharpe: float = 0.5,
    max_drawdown: float = 0.3,
) -> dict[str, Any]:
    """Identify parameter combinations inside a stability region."""
    rows = list(sweep_result.get("results") or [])
    stable = [
        row
        for row in rows
        if float(row.get("sharpe", -np.inf)) >= float(min_sharpe)
        and float(row.get("max_drawdown", np.inf)) <= float(max_drawdown)
    ]
    return {
        "name": "stability_regions",
        "n_stable": len(stable),
        "n_total": len(rows),
        "fraction": float(len(stable) / len(rows)) if rows else 0.0,
        "stable": stable,
        "min_sharpe": float(min_sharpe),
        "max_drawdown": float(max_drawdown),
    }


def overfitting_risk(
    in_sample: Any,
    out_of_sample: Any,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Simple IS vs OOS degradation diagnostics (not a green-light alone)."""
    is_r = as_returns(in_sample)
    oos_r = as_returns(out_of_sample)
    is_s = sharpe_ratio(is_r, periods_per_year=periods_per_year)
    oos_s = sharpe_ratio(oos_r, periods_per_year=periods_per_year)
    degradation = float(is_s - oos_s)
    # Risk score in [0, 1]: larger degradation → higher risk
    risk = float(np.clip(degradation / (abs(is_s) + 1.0), 0.0, 1.0))
    return {
        "name": "overfitting_risk",
        "in_sample_sharpe": is_s,
        "out_of_sample_sharpe": oos_s,
        "degradation": degradation,
        "risk_score": risk,
    }
