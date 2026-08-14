"""Core unit tests for Institutional Statistical Forecasting Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.statistical import (
    StatisticalSettings,
    StatisticalTrainer,
    create_statistical_model,
    list_statistical_models,
)
from iqrp.app.forecasting.statistical.base.fitting import (
    fit_ar_ols,
    fit_arma_css,
    information_criteria,
)
from iqrp.app.forecasting.statistical.base.multivariate import (
    engle_granger,
    fevd,
    granger_causality,
    impulse_response,
    johansen_trace,
)
from iqrp.app.forecasting.statistical.base.processes import (
    simulate_ar,
    simulate_arima,
    simulate_arma,
    simulate_cointegrated_pair,
    simulate_ma,
    simulate_seasonal_arima,
    simulate_var,
    to_frame,
)
from iqrp.app.forecasting.statistical.base.selection import (
    rolling_validation_score,
    select_ar_order,
    select_arima_order,
    select_arma_order,
    select_var_lags,
)
from iqrp.app.forecasting.statistical.base.stationarity import (
    adf_test,
    box_cox,
    difference,
    kpss_test,
    log_transform,
    phillips_perron_test,
    seasonal_difference,
    suggest_differencing,
)
from iqrp.app.forecasting.statistical.diagnostics import run_diagnostics
from iqrp.app.forecasting.statistical.visualization import (
    plot_acf,
    plot_forecast,
    plot_irf,
    plot_qq,
    plot_residuals,
    plot_rolling_comparison,
    plot_seasonal_decomposition,
)


@pytest.mark.unit
def test_registry_lists_all_models() -> None:
    names = set(list_statistical_models())
    assert names >= {
        "ar",
        "ma",
        "arma",
        "arima",
        "sarima",
        "var",
        "varmax",
        "vecm",
        "ses",
        "holt",
        "holt_winters",
    }


@pytest.mark.unit
def test_stationarity_and_transforms() -> None:
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=200))
    assert difference(y, order=1).size == 199
    assert seasonal_difference(y, period=12, order=1).size > 0
    assert adf_test(rng.normal(size=200)).method == "adf"
    assert kpss_test(rng.normal(size=200)).stationary in {True, False}
    assert phillips_perron_test(y).method == "pp"
    assert suggest_differencing(y, max_d=2) >= 1
    assert log_transform(np.abs(rng.normal(size=20)) + 1).size == 20
    assert box_cox(np.abs(rng.normal(size=20)) + 1, lam=0.5).size == 20
    assert box_cox(np.abs(rng.normal(size=20)) + 1, lam=0.0).size == 20


@pytest.mark.unit
def test_selection_and_fitting() -> None:
    y = simulate_ar(180, [0.5, -0.2], rng=np.random.default_rng(1))
    ar = select_ar_order(y, max_p=3)
    assert "p" in ar.best_order
    arma = select_arma_order(y, max_p=2, max_q=2, parallel=True)
    assert "q" in arma.best_order
    arima = select_arima_order(np.cumsum(y), max_p=2, max_d=1, max_q=1, parallel=False)
    assert "d" in arima.best_order
    fit = fit_ar_ols(y, 2)
    assert fit.sigma2 > 0
    fit2 = fit_arma_css(y, 1, 1)
    assert fit2.nobs > 0
    assert "aic" in information_criteria(fit.loglik, fit.k_params, fit.nobs)
    scores = rolling_validation_score(
        y,
        lambda train, h: np.full(h, train[-1]),
        train_size=40,
        horizon=2,
        step=20,
    )
    assert scores["n"] > 0


@pytest.mark.unit
def test_univariate_models_roundtrip(tmp_path: Path) -> None:
    y = simulate_arma(160, [0.4], [0.3], rng=np.random.default_rng(2))
    frame = to_frame(y)
    frame = frame.with_columns(pl.Series("regime", (np.arange(len(y)) > 80).astype(int)))
    settings = StatisticalSettings.from_mapping(
        {
            "identification": {"auto": False},
            "order": {"p": 1, "d": 0, "q": 1, "max_p": 2, "max_q": 2, "max_d": 1},
        }
    )
    for name in ("ar", "ma", "arma", "arima", "ses", "holt"):
        kwargs = {"p": 1} if name in {"ar", "arma", "arima"} else {}
        if name in {"ma", "arma", "arima"}:
            kwargs["q"] = 1
        if name == "arima":
            kwargs["d"] = 0
        model = create_statistical_model(name, settings=settings, **kwargs)
        model.fit(frame, target_column="target", regime_column="regime")
        assert model.is_fitted
        pred = model.predict(frame)
        assert pred.shape[0] == frame.height
        fc = model.forecast(frame, horizon=4)
        assert fc.values.shape[0] == 4
        assert model.forecast_interval(frame, horizon=3)
        assert model.residuals().size
        assert model.diagnostics().nobs
        assert model.evaluate(frame, target_column="target").metrics
        path = tmp_path / f"{name}.json"
        model.save(path)
        loaded = type(model).load(path)
        assert loaded.is_fitted
        model.partial_fit(frame.slice(140, 20), target_column="target")


@pytest.mark.unit
def test_sarima_holt_winters() -> None:
    y = simulate_seasonal_arima(180, period=12, rng=np.random.default_rng(3))
    frame = to_frame(y)
    sarima = create_statistical_model("sarima", seasonal_period=12)
    sarima.fit(frame, target_column="target")
    assert sarima.forecast(frame, horizon=6).values.size == 6
    hw = create_statistical_model("holt_winters", seasonal_period=12)
    hw.fit(frame, target_column="target")
    assert hw.forecast(frame, horizon=6).values.size == 6


@pytest.mark.unit
def test_multivariate_models() -> None:
    coefs = np.array([[[0.5, 0.1], [0.0, 0.4]]])
    Y = simulate_var(150, coefs, rng=np.random.default_rng(4))
    frame = to_frame(Y, prefix="y")
    var = create_statistical_model("var", p=1)
    var.fit(frame, feature_columns=["y0", "y1"], target_column="y0")
    assert var.impulse_response(horizon=5).shape[0] == 5
    assert var.fevd(horizon=5).shape[0] == 5
    assert var.granger(0, 1).lag == 1
    assert select_var_lags(Y, max_lags=3).best_order["p"] >= 1
    # varmax with exog
    frame2 = frame.with_columns(pl.Series("x0", np.linspace(0, 1, frame.height)))
    settings = StatisticalSettings.from_mapping(
        {
            "columns": {
                "endogenous": ("y0", "y1"),
                "exogenous": ("x0",),
                "target": "y0",
                "timestamp": "open_time",
            },
            "identification": {"auto": False},
            "order": {"p": 1},
        }
    )
    vx = create_statistical_model("varmax", settings=settings, p=1, q=1)
    vx.fit(frame2, target_column="y0")
    assert vx.forecast(frame2, horizon=3).values.size == 3
    # vecm
    pair = simulate_cointegrated_pair(160, rng=np.random.default_rng(5))
    f3 = to_frame(pair, prefix="y")
    vecm = create_statistical_model("vecm", lags=1)
    vecm.fit(f3, feature_columns=["y0", "y1"])
    assert "johansen" in vecm.cointegration_test()
    assert vecm.forecast(f3, horizon=4).values.size == 4
    assert johansen_trace(pair).method == "johansen"
    assert engle_granger(pair[:, 0], pair[:, 1]).method == "engle_granger"
    assert granger_causality(Y, cause=0, effect=1).f_stat >= 0
    irf = impulse_response(coefs, np.eye(2), horizon=4)
    assert fevd(coefs, np.eye(2), horizon=4).shape == (4, 2, 2)
    assert irf.shape[0] == 4


@pytest.mark.unit
def test_trainer_and_viz(tmp_path: Path) -> None:
    y = simulate_ar(120, [0.5], rng=np.random.default_rng(6))
    frame = to_frame(y)
    trainer = StatisticalTrainer(
        StatisticalSettings.from_mapping(
            {"identification": {"auto": False}, "order": {"max_p": 2, "max_q": 1, "max_d": 1}}
        )
    )
    model, res = trainer.fit("ar", frame, target_column="target")
    assert res.metrics
    # constrained auto
    model2, res2 = StatisticalTrainer(
        StatisticalSettings.from_mapping({"order": {"max_p": 1, "max_q": 0, "max_d": 0}})
    ).auto_arima(frame)
    assert res2.selection is not None
    board = trainer.compare(["ar", "ses"], frame)
    model = model2  # for viz below
    assert board
    settings = StatisticalSettings.default()
    plot_forecast(y[-40:], model.predict(frame)[-40:], tmp_path / "f.svg", settings=settings)
    plot_residuals(model.residuals(), tmp_path / "r.svg", settings=settings)
    plot_acf(model.diagnostics().acf, tmp_path / "a.svg", settings=settings)
    plot_qq(model.residuals(), tmp_path / "q.svg", settings=settings)
    plot_seasonal_decomposition(y, 12, tmp_path / "s.svg", settings=settings)
    plot_rolling_comparison(
        y[-40:], {"m": model.predict(frame)[-40:]}, tmp_path / "c.svg", settings=settings
    )
    # IRF plot
    coefs = np.array([[[0.4, 0.0], [0.1, 0.3]]])
    plot_irf(impulse_response(coefs, np.eye(2), horizon=6), tmp_path / "i.svg", settings=settings)
    assert run_diagnostics(np.random.default_rng(0).normal(size=80)).ljung_box_pvalue >= 0


@pytest.mark.unit
def test_settings_hydra() -> None:
    s = StatisticalSettings.from_hydra(overrides=["forecast.default_horizon=7"])
    assert s.forecast.default_horizon == 7
    s2 = StatisticalSettings.from_mapping({"online": {"mode": "sliding", "window": 50}})
    assert s2.online.mode == "sliding"
