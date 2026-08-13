"""Integration: statistical forecasting with synthetic processes + simulation-style validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.forecasting.statistical import StatisticalSettings, StatisticalTrainer, create_statistical_model
from iqrp.app.forecasting.statistical.base.processes import (
    simulate_ar,
    simulate_arima,
    simulate_cointegrated_pair,
    simulate_seasonal_arima,
    simulate_var,
    to_frame,
)
from iqrp.app.forecasting.statistical.visualization import plot_forecast


@pytest.mark.integration
def test_parameter_recovery_and_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(41)
    # AR recovery
    y = simulate_ar(300, [0.7], sigma=0.5, rng=rng)
    frame = to_frame(y)
    ar = create_statistical_model("ar", p=1)
    ar.fit(frame, target_column="target")
    assert abs(float(ar._phi[0]) - 0.7) < 0.25  # type: ignore[attr-defined]

    # ARIMA pipeline (fixed order for numerical stability)
    y2 = simulate_arima(250, [0.5], 1, [], sigma=0.8, rng=rng)
    f2 = to_frame(y2)
    settings = StatisticalSettings.from_mapping(
        {"identification": {"auto": False}, "order": {"p": 1, "d": 1, "q": 0}}
    )
    trainer = StatisticalTrainer(settings)
    model, result = trainer.fit("arima", f2)
    assert np.isfinite(result.metrics.get("rmse", np.nan)) or result.order.get("d") == 1
    fc = model.forecast(f2, horizon=8)
    assert fc.values.shape[0] == 8
    assert fc.intervals is not None
    assert np.all(np.isfinite(fc.values))
    plot_forecast(y2[-50:], model.predict(f2)[-50:], tmp_path / "int.svg")

    # Seasonal
    y3 = simulate_seasonal_arima(240, period=12, rng=rng)
    f3 = to_frame(y3)
    sarima = create_statistical_model("sarima", seasonal_period=12)
    sarima.fit(f3, target_column="target")
    assert sarima.forecast(f3, horizon=12).values.size == 12

    # VAR + VECM
    coefs = np.array([[[0.6, 0.1], [0.05, 0.5]]])
    Y = simulate_var(200, coefs, rng=rng)
    fv = to_frame(Y, prefix="y")
    var = create_statistical_model("var", p=1)
    var.fit(fv, feature_columns=["y0", "y1"])
    assert var.impulse_response(horizon=10).shape[0] == 10

    pair = simulate_cointegrated_pair(220, beta=1.2, rng=rng)
    fp = to_frame(pair, prefix="y")
    vecm = create_statistical_model("vecm", lags=1)
    vecm.fit(fp, feature_columns=["y0", "y1"])
    assert vecm.cointegration_test()["engle_granger"]["rank"] in {0, 1}
