"""Tail risk: VaR, CVaR, ES, CTE, drawdown analytics."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.risk.base import RiskMeasure, RiskState
from iqrp.app.risk.tail.cvar import historical_cvar, monte_carlo_cvar, parametric_cvar
from iqrp.app.risk.tail.drawdown import (
    downside_deviation,
    drawdown_series,
    drawdown_state,
    expected_drawdown,
    max_drawdown,
    ulcer_index,
)
from iqrp.app.risk.tail.expected_shortfall import conditional_tail_expectation, expected_shortfall
from iqrp.app.risk.tail.tail_dependence import empirical_tail_dependence
from iqrp.app.risk.tail.var import (
    filtered_historical_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
)


N_SIM = 300
SEED = 42


class TestHistoricalVaR:
    def test_positive_loss(self, returns_1d: np.ndarray) -> None:
        m = historical_var(returns_1d, confidence=0.95)
        assert m.name == "var"
        assert m.value >= 0.0
        assert m.method == "historical"

    def test_horizon_scales(self, returns_1d: np.ndarray) -> None:
        one = historical_var(returns_1d, horizon=1)
        ten = historical_var(returns_1d, horizon=10)
        assert ten.value == pytest.approx(one.value * np.sqrt(10), rel=1e-9)

    def test_confidence_snap(self, returns_1d: np.ndarray) -> None:
        m = historical_var(returns_1d, confidence=0.94)
        assert m.confidence in (0.90, 0.95, 0.99)

    def test_empty(self) -> None:
        assert historical_var([]).value == 0.0


class TestParametricVaR:
    def test_basic(self, returns_1d: np.ndarray) -> None:
        m = parametric_var(returns_1d)
        assert m.method == "parametric"
        assert m.value >= 0.0
        assert "mu" in m.parameters

    def test_single_obs(self) -> None:
        m = parametric_var([0.01])
        assert m.parameters["sigma"] == 0.0

    def test_empty(self) -> None:
        assert parametric_var([]).value == 0.0


class TestMonteCarloVaR:
    def test_reproducible(self, returns_1d: np.ndarray) -> None:
        a = monte_carlo_var(returns_1d, n_simulations=N_SIM, seed=SEED)
        b = monte_carlo_var(returns_1d, n_simulations=N_SIM, seed=SEED)
        assert a.value == b.value
        assert a.method == "monte_carlo"

    def test_clamps_n_sim(self, returns_1d: np.ndarray) -> None:
        m = monte_carlo_var(returns_1d, n_simulations=5, seed=SEED)
        assert m.parameters["n_simulations"] >= 100

    def test_empty(self) -> None:
        assert monte_carlo_var([], n_simulations=N_SIM, seed=SEED).value == 0.0

    def test_horizon(self, returns_1d: np.ndarray) -> None:
        m = monte_carlo_var(returns_1d, horizon=5, n_simulations=N_SIM, seed=SEED)
        assert m.horizon == 5
        assert m.value >= 0.0


class TestFilteredHistoricalVaR:
    def test_basic(self, returns_1d: np.ndarray) -> None:
        m = filtered_historical_var(returns_1d, lambda_=0.94)
        assert m.method == "filtered_historical"
        assert m.value >= 0.0
        assert "latest_vol" in m.parameters

    def test_short_series(self) -> None:
        assert filtered_historical_var([0.01]).value == 0.0
        assert filtered_historical_var([]).value == 0.0

    def test_lambda_clip(self, returns_1d: np.ndarray) -> None:
        m = filtered_historical_var(returns_1d, lambda_=2.0)
        assert 0.0 < m.parameters["lambda"] < 1.0


class TestCVaR:
    def test_historical(self, returns_1d: np.ndarray) -> None:
        var = historical_var(returns_1d)
        cvar = historical_cvar(returns_1d)
        assert cvar.name == "cvar"
        assert cvar.value >= var.value - 1e-9

    def test_parametric(self, returns_1d: np.ndarray) -> None:
        m = parametric_cvar(returns_1d, confidence=0.99)
        assert m.method == "parametric"
        assert m.value >= 0.0

    def test_monte_carlo(self, returns_1d: np.ndarray) -> None:
        m = monte_carlo_cvar(returns_1d, n_simulations=N_SIM, seed=SEED)
        assert m.method == "monte_carlo"
        assert m.value >= 0.0

    def test_empty_all(self) -> None:
        assert historical_cvar([]).value == 0.0
        assert parametric_cvar([]).value == 0.0
        assert monte_carlo_cvar([], n_simulations=N_SIM, seed=SEED).value == 0.0


class TestExpectedShortfallAndCTE:
    def test_historical_es(self, returns_1d: np.ndarray) -> None:
        m = expected_shortfall(returns_1d, method="historical")
        assert m.name == "expected_shortfall"
        assert m.value == pytest.approx(historical_cvar(returns_1d).value)

    def test_parametric_aliases(self, returns_1d: np.ndarray) -> None:
        for method in ("parametric", "gaussian", "normal"):
            m = expected_shortfall(returns_1d, method=method)
            assert m.method == "parametric"

    def test_mc_aliases(self, returns_1d: np.ndarray) -> None:
        for method in ("monte_carlo", "mc", "simulation"):
            m = expected_shortfall(
                returns_1d, method=method, n_simulations=N_SIM, seed=SEED
            )
            assert m.method == "monte_carlo"

    def test_cte_quantile(self, returns_1d: np.ndarray) -> None:
        m = conditional_tail_expectation(returns_1d, confidence=0.95)
        assert m.name == "conditional_tail_expectation"
        assert m.value >= 0.0

    def test_cte_absolute_threshold(self, returns_1d: np.ndarray) -> None:
        m = conditional_tail_expectation(returns_1d, threshold=-0.02)
        assert m.value >= 0.0

    def test_cte_empty(self) -> None:
        assert conditional_tail_expectation([]).value == 0.0


class TestDrawdown:
    def test_series_shape(self, returns_1d: np.ndarray) -> None:
        dd = drawdown_series(returns_1d)
        assert dd.shape == returns_1d.shape
        assert np.all(dd >= -1e-12)

    def test_empty_series(self) -> None:
        dd = drawdown_series([])
        assert dd.size == 0

    def test_max_drawdown(self, returns_1d: np.ndarray) -> None:
        m = max_drawdown(returns_1d)
        assert m.name == "max_drawdown"
        assert 0.0 <= m.value <= 1.0

    def test_expected_drawdown(self, returns_1d: np.ndarray) -> None:
        assert expected_drawdown(returns_1d).value >= 0.0

    def test_ulcer(self, returns_1d: np.ndarray) -> None:
        assert ulcer_index(returns_1d).value >= 0.0

    def test_downside_deviation(self, returns_1d: np.ndarray) -> None:
        m = downside_deviation(returns_1d, mar=0.0)
        assert m.name == "downside_deviation"
        assert m.value >= 0.0

    def test_empty_measures(self) -> None:
        assert max_drawdown([]).value == 0.0
        assert expected_drawdown([]).value == 0.0
        assert ulcer_index([]).value == 0.0
        assert downside_deviation([]).value == 0.0

    def test_known_drawdown_path(self) -> None:
        # Up then sharp down
        r = np.array([0.10, 0.10, -0.20, -0.10, 0.05])
        dd = drawdown_series(r)
        assert dd[-2] > 0.0
        assert max_drawdown(r).value == pytest.approx(float(np.max(dd)))

    def test_drawdown_state_normal(self, returns_1d: np.ndarray) -> None:
        st = drawdown_state(returns_1d)
        assert st["risk_state"] in {s.value for s in RiskState}
        assert "current_drawdown" in st
        assert "max_drawdown" in st
        assert "measures" in st
        assert "thresholds" in st

    def test_drawdown_state_halt(self) -> None:
        crash = np.full(40, -0.03)
        st = drawdown_state(crash, trading_halt=0.20)
        assert st["risk_state"] == RiskState.TRADING_HALT.value
        assert st["current_drawdown"] >= 0.20 - 1e-9

    def test_drawdown_state_levels(self) -> None:
        # Construct controlled current drawdown by single drop
        for thr, expected in [
            (0.04, RiskState.NORMAL),
            (0.06, RiskState.CAUTION),
            (0.11, RiskState.REDUCED_RISK),
            (0.16, RiskState.CAPITAL_PRESERVATION),
            (0.25, RiskState.TRADING_HALT),
        ]:
            r = np.array([0.0, -thr])
            st = drawdown_state(
                r,
                caution=0.05,
                reduced_risk=0.10,
                capital_preservation=0.15,
                trading_halt=0.20,
            )
            assert st["risk_state"] == expected.value, (thr, st)

    def test_drawdown_state_empty(self) -> None:
        st = drawdown_state([])
        assert st["risk_state"] == RiskState.NORMAL.value
        assert st["current_drawdown"] == 0.0

    def test_recovery_time_present(self) -> None:
        r = np.array([0.05, -0.10, -0.05, 0.20, 0.01])
        st = drawdown_state(r)
        assert "recovery_time" in st
        assert "drawdown_duration" in st


class TestTailDependence:
    def test_lower_tail(self, rng: np.random.Generator) -> None:
        x = rng.normal(0, 1, 200)
        y = 0.8 * x + 0.2 * rng.normal(0, 1, 200)
        m = empirical_tail_dependence(x, y, quantile=0.05, tail="lower")
        assert m.name == "empirical_tail_dependence"
        assert 0.0 <= m.value <= 1.0

    def test_upper_tail(self, rng: np.random.Generator) -> None:
        x = rng.normal(0, 1, 200)
        y = x + rng.normal(0, 0.1, 200)
        m = empirical_tail_dependence(x, y, tail="upper")
        assert m.value >= 0.0

    def test_short_series(self) -> None:
        assert empirical_tail_dependence([1, 2, 3], [1, 2, 3]).value == 0.0

    def test_nan_handling(self) -> None:
        x = np.array([0.01, np.nan, 0.02, -0.03] + [0.0] * 20)
        y = np.array([-0.01, 0.02, np.inf, 0.01] + [0.0] * 20)
        m = empirical_tail_dependence(x, y)
        assert isinstance(m, RiskMeasure)
        assert np.isfinite(m.value)
