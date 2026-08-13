"""Integration tests: volatility engine + forecasting framework + simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.volatility import (
    VolatilitySettings,
    VolatilityTrainer,
    create_volatility_model,
    ensure_volatility_models_loaded,
)
from iqrp.app.forecasting.volatility.base.processes import simulate_dcc, simulate_garch, to_returns_frame


@pytest.mark.integration
def test_end_to_end_risk_pipeline(tmp_path: Path) -> None:
    ensure_volatility_models_loaded()
    assert "garch" in get_registry().list_names()

    r, _ = simulate_garch(300, rng=np.random.default_rng(21))
    frame = to_returns_frame(r)
    settings = VolatilitySettings.from_hydra(
        overrides=["distribution.name=gaussian", "online.mode=rolling", "online.window=120"]
    )
    trainer = VolatilityTrainer(settings)
    model, result = trainer.auto_select(frame, candidates=["ewma", "garch", "gjr_garch"])
    assert result.selection is not None

    # risk outputs
    sigma_ann = model.annualized_volatility()
    var1 = model.forecast_covariance(horizon=1)
    fc = model.forecast(frame, horizon=5)
    assert sigma_ann.mean() > 0
    assert var1.size >= 1
    assert fc.intervals is not None

    # online update
    new_r, _ = simulate_garch(40, rng=np.random.default_rng(22))
    model.partial_fit(to_returns_frame(new_r))

    path = tmp_path / "vol.pkl"
    model.save(path)
    loaded = create_volatility_model(model.meta.name)
    loaded = type(model).load(path)
    assert loaded.evaluate(frame).metrics

    # multivariate portfolio cov
    rets, _ = simulate_dcc(120, rng=np.random.default_rng(23))
    mframe = to_returns_frame(rets, prefix="x")
    dcc = create_volatility_model("dcc_garch", settings=settings)
    dcc.fit(mframe, feature_columns=["x0", "x1"])
    cov = dcc.forecast_covariance(horizon=5)
    assert cov.shape == (5, 2, 2)


@pytest.mark.integration
def test_regime_conditioned_forecast() -> None:
    r, _ = simulate_garch(220, rng=np.random.default_rng(24))
    regime = np.where(np.abs(r) > np.median(np.abs(r)), 1, 0)
    frame = to_returns_frame(r, regime=regime)
    settings = VolatilitySettings.from_mapping(
        {"regime": {"enabled": True, "condition": True, "ensemble_weight": True}}
    )
    model = create_volatility_model("garch", settings=settings)
    model.fit(frame, regime_column="regime")
    fc = model.forecast(frame, horizon=3)
    assert fc.path().size == 3
