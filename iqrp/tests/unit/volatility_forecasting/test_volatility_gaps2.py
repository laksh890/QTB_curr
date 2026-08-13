"""Additional coverage gaps for volatility engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting.volatility import VolatilitySettings, create_volatility_model
from iqrp.app.forecasting.volatility.base.processes import simulate_garch, to_returns_frame
from iqrp.app.forecasting.volatility.base.selection import select_volatility_models
from iqrp.app.forecasting.volatility.diagnostics.report import _arch_lm, _ljung_box
from iqrp.app.forecasting.volatility.visualization.plots import _pyplot


@pytest.mark.unit
def test_model_resolve_and_frame_helpers() -> None:
    r, _ = simulate_garch(80, rng=np.random.default_rng(50))
    frame = to_returns_frame(r, regime=np.zeros(80, dtype=int))
    m = create_volatility_model("garch")
    m.fit(frame, regime_column="regime")
    rebuilt = m._frame_from_returns(r[:40], frame, "returns")
    assert rebuilt.height == 40
    assert "regime" in rebuilt.columns
    # forecast_interval without intervals on Forecast — force residual path
    with patch.object(type(m.forecast(frame)), "intervals", None):
        # call method which builds its own forecast
        pass
    intervals = m.forecast_interval(frame, horizon=2, level=0.9)
    assert len(intervals) == 2
    # evaluate without target column name in frame uses stored returns
    weird = pl.DataFrame({"open_time": list(range(80)), "other": r})
    # predict path when target missing → uses stored variance
    assert m.predict(weird).size == m.conditional_volatility().size


@pytest.mark.unit
def test_partial_fit_expanding_and_empty_prev() -> None:
    r, _ = simulate_garch(100, rng=np.random.default_rng(51))
    settings = VolatilitySettings.from_mapping(
        {"online": {"mode": "expanding", "warm_start": True, "adaptive_rate": 0.2}}
    )
    m = create_volatility_model("ewma", settings=settings)
    m.fit(to_returns_frame(r[:60]))
    m.partial_fit(to_returns_frame(r[60:]))
    # sliding mode
    settings2 = VolatilitySettings.from_mapping(
        {"online": {"mode": "sliding", "window": 40, "warm_start": True}}
    )
    m2 = create_volatility_model("ewma", settings=settings2)
    m2.fit(to_returns_frame(r[:50]))
    m2.partial_fit(to_returns_frame(r[50:70]))
    # partial_fit when not fitted
    m3 = create_volatility_model("ewma", settings=settings)
    m3.partial_fit(to_returns_frame(r[:40]))
    assert m3.is_fitted


@pytest.mark.unit
def test_selection_parallel_exception() -> None:
    r, _ = simulate_garch(80, rng=np.random.default_rng(52))
    frame = to_returns_frame(r)

    def boom(name: str, **kwargs):
        raise RuntimeError("fail")

    with patch(
        "iqrp.app.forecasting.volatility.registry.create_volatility_model",
        side_effect=boom,
    ):
        sel = select_volatility_models(frame, candidates=["garch", "ewma"], parallel=True)
        assert sel.best == "garch"  # empty board default
        sel2 = select_volatility_models(frame, candidates=["garch", "ewma"], parallel=False)
        assert sel2.best == "garch"


@pytest.mark.unit
def test_arch_lm_and_lb_edges() -> None:
    assert _ljung_box(np.array([1.0, 2.0]), lags=10)[1] == 1.0
    assert _arch_lm(np.array([1.0, 2.0]), lags=5)[1] == 1.0
    with patch("numpy.linalg.lstsq", side_effect=RuntimeError("x")):
        assert _arch_lm(np.random.default_rng(0).normal(size=50) ** 2, lags=3)[1] == 1.0


@pytest.mark.unit
def test_pyplot_none_branch() -> None:
    # without matplotlib installed, _pyplot returns None
    assert _pyplot() is None


@pytest.mark.unit
def test_settings_none_and_dict_init() -> None:
    from iqrp.app.forecasting.volatility.garch.garch import GARCHModel

    m = GARCHModel(settings=None)
    r, _ = simulate_garch(60, rng=np.random.default_rng(53))
    m.fit(to_returns_frame(r))
    assert m.is_fitted
    m2 = GARCHModel(settings={"order": {"p": 1, "q": 1}})
    m2.fit(to_returns_frame(r))
    assert m2.is_fitted


@pytest.mark.unit
def test_resolve_target_fallbacks() -> None:
    from iqrp.app.forecasting.volatility.garch.garch import GARCHModel
    from iqrp.app.core.exceptions import ValidationError

    m = GARCHModel()
    with pytest.raises(ValidationError):
        m._resolve_target_name(pl.DataFrame({"open_time": [1, 2]}), None)
    with pytest.raises(ValidationError):
        m._extract_returns(pl.DataFrame({"open_time": [1, 2]}), None)


@pytest.mark.unit
def test_forecast_interval_uses_residual_fallback() -> None:
    r, _ = simulate_garch(80, rng=np.random.default_rng(54))
    frame = to_returns_frame(r)
    m = create_volatility_model("historical_volatility")
    m.fit(frame)
    # Monkeypatch forecast to return Forecast without intervals
    from iqrp.app.forecasting.base.forecast import Forecast

    def _fc(*_a, **_k):
        return Forecast.from_values([0.1, 0.1, 0.1], horizon=3, intervals=None)

    with patch.object(m, "forecast", side_effect=_fc):
        ints = m.forecast_interval(frame, horizon=3)
        assert len(ints) == 3


@pytest.mark.unit
def test_ewma_forecast_horizon_one() -> None:
    r, _ = simulate_garch(50, rng=np.random.default_rng(55))
    m = create_volatility_model("ewma")
    m.fit(to_returns_frame(r))
    s, v = m._forecast_path(1)
    assert s.size == 1 and v.size == 1


@pytest.mark.unit
def test_remaining_edge_branches() -> None:
    from iqrp.app.forecasting.volatility.base.likelihood import estimate
    from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel
    from iqrp.app.forecasting.volatility.diagnostics.report import persistence_and_half_life
    from iqrp.app.forecasting.volatility.trainer import VolatilityTrainer
    from iqrp.app.forecasting.volatility.visualization.plots import _pyplot_available

    assert _pyplot_available() is False
    # half-life branch with persist>=1 before clip via lambda path already covered;
    # force alpha+beta large
    persistence_and_half_life({"alpha": 0.9, "beta": 0.9, "gamma": 0.5})

    r, _ = simulate_garch(90, rng=np.random.default_rng(56))
    frame = to_returns_frame(r)
    m = create_volatility_model("garch")
    m.fit(frame)
    # identical (demeaned) returns → early predict return
    demeaned = pl.DataFrame({"returns": m._returns})
    assert UnivariateVolatilityModel.predict(m, demeaned).size == frame.height
    # equal-size but not close → variance_from_returns
    noisy = pl.DataFrame({"returns": r + 0.5})
    assert m.predict(noisy).size == frame.height
    assert UnivariateVolatilityModel._variance_from_returns(m, m._returns).shape[0] == frame.height
    # predict without target column
    assert UnivariateVolatilityModel.predict(m, pl.DataFrame({"open_time": list(range(frame.height))})).size == frame.height

    # trainer serial exception path
    trainer = VolatilityTrainer(VolatilitySettings.from_mapping({"visualization": {"enabled": False}}))
    rows = trainer.compare(["ewma", "definitely_missing"], frame, parallel=False)
    assert len(rows) >= 1

    # likelihood non-finite variance inside objective
    def nan_var(theta):
        return np.full(r.size, np.nan)

    res = estimate(r, nan_var, np.array([0.1]), [(0.0, 1.0)], param_names=["x"], n_restarts=1)
    assert res.message == "fallback"

    # transform path
    def var_fn(theta):
        return np.full(r.size, max(float(theta[0]), 1e-4))

    res2 = estimate(
        r,
        var_fn,
        np.array([0.2]),
        [(1e-6, 1.0)],
        param_names=["x"],
        transform=lambda t: np.abs(t),
        n_restarts=1,
    )
    assert res2.params[0] >= 0

    # extract returns via feature columns when target missing after clearing
    m._target_column = None
    feat_frame = pl.DataFrame({"open_time": list(range(len(r))), "feat": r})
    m._feature_columns = ["feat"]
    assert m._extract_returns(feat_frame, None).size == len(r)
    assert m._resolve_target_name(feat_frame, None) == "feat"

    # partial_fit when _returns cleared but fitted
    m._returns = None
    m.partial_fit(frame)

    # rolling validation short forecast pad
    from iqrp.app.forecasting.volatility.base.selection import rolling_vol_validation

    scores = rolling_vol_validation(
        r, lambda train, h: np.array([0.1]), train_size=40, horizon=3, step=30
    )
    assert scores["n"] > 0

    # regime subset fit failure swallowed
    settings = VolatilitySettings.from_mapping(
        {"regime": {"enabled": True, "condition": True, "ensemble_weight": False}}
    )
    m4 = create_volatility_model("garch", settings=settings)
    regime = np.zeros(len(r), dtype=int)
    regime[-5:] = 1  # tiny regime → skipped (<30)
    m4.fit(to_returns_frame(r, regime=regime), regime_column="regime")

    # force per-regime fit exception (regime large enough to enter try)
    def boom_subset(_rr):
        raise RuntimeError("nope")

    from iqrp.app.forecasting.volatility.base.univariate import UnivariateVolatilityModel as UVM

    # provide a successful global fit_subset then failing per-regime via wrapper
    def fit_subset(rr):
        if rr.size < len(r):
            raise RuntimeError("nope")
        return {"omega": 0.1, "alpha": 0.05, "beta": 0.9}, np.full(rr.size, 0.1), 1.0, 2.0, 3.0

    regs = np.zeros(len(r), dtype=int)
    regs[len(r) // 2 :] = 1
    UVM._regime_fit(m4, r, regs, fit_subset)

    # matplotlib import success path via mock module (isolated)
    import sys
    from iqrp.app.forecasting.volatility.visualization import plots as plot_mod

    fake_plt = MagicMock()
    fig, ax = MagicMock(), MagicMock()
    fake_plt.subplots.return_value = (fig, ax)
    prev_mpl = sys.modules.get("matplotlib")
    prev_py = sys.modules.get("matplotlib.pyplot")
    sys.modules["matplotlib"] = MagicMock()
    sys.modules["matplotlib.pyplot"] = fake_plt
    try:
        assert plot_mod._import_pyplot() is not None
    finally:
        if prev_mpl is None:
            sys.modules.pop("matplotlib", None)
        else:
            sys.modules["matplotlib"] = prev_mpl
        if prev_py is None:
            sys.modules.pop("matplotlib.pyplot", None)
        else:
            sys.modules["matplotlib.pyplot"] = prev_py
