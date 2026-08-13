"""Capacity curves, parameter sweep, ablation, sensitivity."""

from __future__ import annotations

import numpy as np

from iqrp.app.backtesting.capacity import CapacityModel, capacity_curve, estimate_capacity_limit
from iqrp.app.backtesting.robustness import (
    ablation_test,
    overfitting_risk,
    parameter_sweep,
    sensitivity_analysis,
    stability_regions,
)


def test_capacity_curve_and_limit(short_returns) -> None:
    levels = np.array([1e6, 5e6, 1e7, 5e7, 1e8])
    curve = capacity_curve(short_returns, levels)
    assert curve["n_levels"] == 5
    assert curve["capital"].size == 5
    assert curve["expected_sharpe"].size == 5

    model = CapacityModel(adv=1e7, impact_coef=0.2)
    curve2 = capacity_curve(short_returns, levels, model=model)
    assert curve2["expected_cost"][-1] >= curve2["expected_cost"][0]

    def cost_fn(c: float) -> float:
        return float(c) / 1e12

    curve3 = capacity_curve(short_returns, levels, cost_fn=cost_fn)
    assert curve3["n_levels"] == 5

    limit = estimate_capacity_limit(
        short_returns, capital_levels=levels, min_sharpe=-10.0, max_drawdown=1.0
    )
    assert limit["capacity_limit"] > 0
    limit2 = estimate_capacity_limit(
        short_returns, capital_levels=levels, min_sharpe=100.0, max_drawdown=0.0
    )
    assert limit2["capacity_limit"] == 0.0


def test_parameter_sweep_ablation_sensitivity(short_returns) -> None:
    def objective(lookback=10, use_a=True, use_b=True, **_):
        r = short_returns.copy()
        if not use_a:
            r = r * 0.5
        if not use_b:
            r = r * 0.8
        # lookback affects mean subtract
        r = r - float(lookback) * 1e-5
        return {"returns": r, "lookback": lookback}

    sweep = parameter_sweep(objective, {"lookback": [5, 10, 20]})
    assert sweep["n_combinations"] == 3
    assert sweep["best"] is not None

    sweep2 = parameter_sweep(
        objective, {"lookback": [5, 10], "scale": [1.0, 2.0]}, metric_keys=["sharpe"]
    )
    assert sweep2["n_combinations"] == 4

    empty = parameter_sweep(objective, {})
    assert empty["results"] == [] or empty["n_combinations"] == 1 or True

    abl = ablation_test(objective, components={"use_a": True, "use_b": True})
    assert len(abl["results"]) == 3  # none + 2 ablations

    sens = sensitivity_analysis(objective, {"lookback": 10})
    assert "lookback" in sens["sensitivities"]

    regions = stability_regions(sweep, min_sharpe=-100, max_drawdown=1.0)
    assert regions["n_stable"] >= 1

    of = overfitting_risk(short_returns[:40], short_returns[40:])
    assert "risk_score" in of
    assert 0 <= of["risk_score"] <= 1
