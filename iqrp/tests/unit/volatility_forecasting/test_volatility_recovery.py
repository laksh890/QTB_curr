"""Synthetic GARCH parameter recovery and stress tests."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.forecasting.volatility import create_volatility_model
from iqrp.app.forecasting.volatility.base.processes import simulate_garch, to_returns_frame
from iqrp.app.forecasting.volatility.evaluation.metrics import qlike


@pytest.mark.unit
def test_garch_parameter_recovery() -> None:
    true_omega, true_alpha, true_beta = 0.05, 0.1, 0.85
    r, true_var = simulate_garch(
        800,
        omega=true_omega,
        alpha=true_alpha,
        beta=true_beta,
        rng=np.random.default_rng(42),
    )
    frame = to_returns_frame(r)
    model = create_volatility_model("garch")
    model.fit(frame)
    alpha = model.params.get("alpha", model.params.get("alpha_0", 0.0))
    beta = model.params.get("beta", model.params.get("beta_0", 0.0))
    # recovery: persistence is the stable estimand; individual α/β trade off
    assert abs((alpha + beta) - (true_alpha + true_beta)) < 0.15
    assert 0.0 < alpha < 0.4
    assert 0.5 < beta < 0.99
    hat = model.conditional_variance()
    assert qlike(true_var, hat) < qlike(true_var, np.full_like(true_var, np.mean(r**2)))


@pytest.mark.unit
def test_ewma_stress_large() -> None:
    r, _ = simulate_garch(2500, rng=np.random.default_rng(5))
    frame = to_returns_frame(r)
    model = create_volatility_model("ewma")
    model.fit(frame)
    assert model.conditional_volatility().size == 2500
    fc = model.forecast(frame, horizon=20)
    assert fc.path().size == 20


@pytest.mark.unit
def test_persistence_half_life() -> None:
    r, _ = simulate_garch(400, alpha=0.05, beta=0.9, rng=np.random.default_rng(9))
    model = create_volatility_model("garch")
    model.fit(to_returns_frame(r))
    d = model.diagnostics()
    assert 0.8 < d.persistence < 1.0
    assert d.half_life > 1.0
