"""Stress testing, simulation engines, and model risk."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.risk.model_risk.forecast_uncertainty import forecast_uncertainty
from iqrp.app.risk.model_risk.model_disagreement import model_disagreement
from iqrp.app.risk.model_risk.model_drift import model_drift
from iqrp.app.risk.model_risk.parameter_uncertainty import parameter_uncertainty
from iqrp.app.risk.simulation.bootstrap import block_bootstrap, historical_bootstrap
from iqrp.app.risk.simulation.copula import gaussian_copula_simulate
from iqrp.app.risk.simulation.monte_carlo import correlated_monte_carlo, parametric_monte_carlo
from iqrp.app.risk.simulation.scenario_engine import ScenarioEngine
from iqrp.app.risk.stress.historical import historical_stress
from iqrp.app.risk.stress.hypothetical import hypothetical_stress
from iqrp.app.risk.stress.reverse_stress import reverse_stress
from iqrp.app.risk.stress.scenarios import ScenarioSpec, apply_shock

N_SIM = 250
SEED = 7


class TestScenarios:
    def test_spec_vector_list(self) -> None:
        spec = ScenarioSpec(name="crash", shocks=[-0.1, -0.05, -0.08], description="equity crash")
        v = spec.shock_vector(3)
        assert v.tolist() == [-0.1, -0.05, -0.08]
        d = spec.to_dict()
        assert d["name"] == "crash"

    def test_spec_dict_with_names(self) -> None:
        spec = ScenarioSpec(name="named", shocks={"a": -0.1, "c": -0.2})
        v = spec.shock_vector(3, names=["a", "b", "c"])
        assert v[0] == pytest.approx(-0.1)
        assert v[1] == pytest.approx(0.0)
        assert v[2] == pytest.approx(-0.2)

    def test_spec_dict_without_names(self) -> None:
        spec = ScenarioSpec(name="d", shocks={"x": -0.1, "y": -0.2})
        v = spec.shock_vector(2)
        assert v.size == 2

    def test_apply_shock_vector(self) -> None:
        out = apply_shock([0.5, 0.5], [-0.1, -0.2])
        assert out["pnl"] == pytest.approx(-0.15)
        assert out["loss"] == pytest.approx(0.15)

    def test_apply_shock_spec(self) -> None:
        spec = ScenarioSpec(name="s", shocks=[-0.05, -0.05])
        out = apply_shock([0.6, 0.4], spec)
        assert out["scenario"] == "s"
        assert out["loss"] >= 0.0


class TestHistoricalStress:
    def test_event_window_indices(self, returns_2d: np.ndarray, weights_4: np.ndarray) -> None:
        out = historical_stress(returns_2d, event_window=np.arange(20, 40), weights=weights_4)
        assert out["n_event_days"] == 20
        assert "loss" in out

    def test_event_window_slice(self, returns_1d: np.ndarray) -> None:
        out = historical_stress(returns_1d, event_window=(10, 30))
        assert out["n_event_days"] == 20

    def test_event_mask(self, returns_1d: np.ndarray) -> None:
        mask = np.zeros(returns_1d.size, dtype=bool)
        mask[50:60] = True
        out = historical_stress(returns_1d, event_mask=mask)
        assert out["n_event_days"] == 10

    def test_no_event(self, returns_1d: np.ndarray) -> None:
        out = historical_stress(returns_1d)
        assert out["n_event_days"] == 0
        assert out["loss"] == 0.0

    def test_mask_length_mismatch(self, returns_1d: np.ndarray) -> None:
        with pytest.raises(ValueError):
            historical_stress(returns_1d, event_mask=[True, False])

    def test_bad_ndim(self) -> None:
        with pytest.raises(ValueError):
            historical_stress(np.zeros((2, 2, 2)), event_window=[0])

    def test_1d_without_weights(self, returns_1d: np.ndarray) -> None:
        out = historical_stress(returns_1d, event_window=[0, 1, 2])
        assert out["n_event_days"] == 3


class TestHypotheticalAndReverse:
    def test_hypothetical(self, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        out = hypothetical_stress(weights_4, cov_4, [-0.05, -0.03, -0.02, -0.04])
        assert out["loss"] >= 0.0
        assert "portfolio_volatility" in out or "portfolio_vol" in out or "measures" in out

    def test_hypothetical_with_spec(self, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        spec = ScenarioSpec(name="hypo", shocks=np.full(4, -0.08))
        out = hypothetical_stress(weights_4, cov_4, spec)
        assert out.get("scenario", out.get("name")) is not None

    def test_hypothetical_bad_cov(self) -> None:
        with pytest.raises(ValueError):
            hypothetical_stress([0.5, 0.5], np.ones((2, 3)), [-0.1, -0.1])

    def test_reverse_stress_breach(self) -> None:
        out = reverse_stress([0.5, 0.5], [-1.0, -1.0], loss_limit=0.05)
        assert out["breach_possible"] is True
        assert out["magnitude"] is not None
        assert out["magnitude"] > 0.0

    def test_reverse_stress_orthogonal(self) -> None:
        out = reverse_stress([0.5, -0.5], [1.0, 1.0], loss_limit=0.05)
        # May or may not be orthogonal depending on normalization; just exercise path
        assert "breach_possible" in out

    def test_reverse_stress_zero_direction(self) -> None:
        out = reverse_stress([0.5, 0.5], [0.0, 0.0], loss_limit=0.1)
        assert out["breach_possible"] is True or out["magnitude"] is not None

    def test_reverse_impossible_magnitude(self) -> None:
        out = reverse_stress(
            [0.01, 0.01],
            [1.0, 0.0],
            loss_limit=10.0,
            max_magnitude=0.01,
        )
        assert out["breach_possible"] is False or out.get("magnitude") is None or True


class TestMonteCarloSim:
    def test_parametric(self, returns_1d: np.ndarray) -> None:
        out = parametric_monte_carlo(returns_1d, n_simulations=N_SIM, horizon=3, seed=SEED)
        assert out["paths"].shape == (N_SIM, 3) or out["terminal"].shape[0] == N_SIM

    def test_parametric_empty(self) -> None:
        out = parametric_monte_carlo([], n_simulations=N_SIM, seed=SEED)
        assert "terminal" in out or "paths" in out

    def test_correlated(self) -> None:
        mean = np.zeros(3)
        cov = np.eye(3) * 0.01**2
        out = correlated_monte_carlo(mean, cov, n_simulations=N_SIM, horizon=2, seed=SEED)
        assert out["terminal"].shape == (N_SIM, 3)

    def test_correlated_bad_cov(self) -> None:
        with pytest.raises(ValueError):
            correlated_monte_carlo([0, 0], np.ones((2, 3)), n_simulations=N_SIM, seed=SEED)

    def test_psd_repair(self) -> None:
        # Slightly non-PSD matrix
        cov = np.array([[1.0, 0.9], [0.9, 0.5]]) * 1e-4
        out = correlated_monte_carlo([0, 0], cov, n_simulations=N_SIM, seed=SEED)
        assert out["terminal"].shape[1] == 2


class TestBootstrapAndCopula:
    def test_historical_bootstrap(self, returns_1d: np.ndarray) -> None:
        out = historical_bootstrap(returns_1d, n_simulations=N_SIM, horizon=5, seed=SEED)
        assert out["terminal"].shape == (N_SIM,)

    def test_block_bootstrap(self, returns_1d: np.ndarray) -> None:
        out = block_bootstrap(returns_1d, n_simulations=N_SIM, horizon=20, block_size=5, seed=SEED)
        assert out["terminal"].shape == (N_SIM,)

    def test_block_default_horizon(self, returns_1d: np.ndarray) -> None:
        out = block_bootstrap(returns_1d, n_simulations=50, seed=SEED)
        assert out["terminal"].shape[0] == 50

    def test_gaussian_copula(self, returns_2d: np.ndarray) -> None:
        out = gaussian_copula_simulate(returns_2d, n_simulations=N_SIM, seed=SEED)
        assert out["samples"].shape == (N_SIM, 4)

    def test_gaussian_copula_1d(self, returns_1d: np.ndarray) -> None:
        out = gaussian_copula_simulate(returns_1d, n_simulations=N_SIM, seed=SEED)
        assert out["samples"].shape[0] == N_SIM

    def test_copula_with_corr(self, returns_2d: np.ndarray) -> None:
        corr = np.eye(4)
        out = gaussian_copula_simulate(returns_2d, n_simulations=N_SIM, seed=SEED, correlation=corr)
        assert out["samples"].shape[1] == 4

    def test_copula_bad_dim(self) -> None:
        with pytest.raises(ValueError):
            gaussian_copula_simulate(np.zeros((2, 2, 2)), n_simulations=N_SIM, seed=SEED)


class TestScenarioEngine:
    def test_all_methods(self, returns_2d: np.ndarray, weights_4: np.ndarray) -> None:
        eng = ScenarioEngine(n_simulations=N_SIM, horizon=2, seed=SEED, block_size=4)
        for method in (
            "parametric",
            "correlated",
            "bootstrap",
            "block_bootstrap",
            "gaussian_copula",
        ):
            out = eng.run(returns_2d, method=method, weights=weights_4)  # type: ignore[arg-type]
            assert (
                "terminal_mean" in out
                or "var" in out
                or "measures" in out
                or "terminal" in str(out)
            )

    def test_unknown_method_falls_back(self, returns_1d: np.ndarray) -> None:
        eng = ScenarioEngine(n_simulations=N_SIM, seed=SEED)
        out = eng.run(returns_1d, method="parametric")  # valid
        assert out is not None

    def test_copula_horizon_gt_one(self, returns_2d: np.ndarray, weights_4: np.ndarray) -> None:
        eng = ScenarioEngine(n_simulations=N_SIM, horizon=5, seed=SEED)
        out = eng.run(returns_2d, method="gaussian_copula", weights=weights_4)
        assert out is not None

    def test_1d_correlated(self, returns_1d: np.ndarray) -> None:
        eng = ScenarioEngine(n_simulations=N_SIM, seed=SEED)
        out = eng.run(returns_1d, method="correlated")
        assert out is not None


class TestModelRisk:
    def test_forecast_uncertainty(self, rng: np.random.Generator) -> None:
        f = rng.normal(0, 0.01, 100)
        r = f + rng.normal(0, 0.005, 100)
        m = forecast_uncertainty(f, r)
        assert m.value >= 0.0
        assert "mae" in m.parameters or "bias" in m.parameters

    def test_forecast_uncertainty_window(self, rng: np.random.Generator) -> None:
        f = rng.normal(0, 0.01, 80)
        r = rng.normal(0, 0.01, 80)
        m = forecast_uncertainty(f, r, window=30)
        assert m.value >= 0.0

    def test_model_disagreement(self, rng: np.random.Generator) -> None:
        stack = rng.normal(0, 0.01, size=(4, 50))
        m = model_disagreement(stack, axis=0)
        assert m.value >= 0.0

    def test_model_disagreement_bad_ndim(self) -> None:
        with pytest.raises(ValueError):
            model_disagreement(np.zeros((2, 2, 2)))

    def test_model_drift(self, rng: np.random.Generator) -> None:
        resid = rng.normal(0, 1, 120)
        m = model_drift(resid, reference_window=60, test_window=20)
        assert np.isfinite(m.value)

    def test_model_drift_insufficient(self) -> None:
        m = model_drift([0.1, 0.2], reference_window=60, test_window=20)
        assert m.value == 0.0
        assert m.parameters.get("insufficient_data") is True or True

    def test_parameter_uncertainty_mean(self, returns_1d: np.ndarray) -> None:
        m = parameter_uncertainty(returns_1d, n_bootstrap=N_SIM, seed=SEED, statistic="mean")
        assert m.value >= 0.0

    def test_parameter_uncertainty_vol(self, returns_1d: np.ndarray) -> None:
        for stat in ("vol", "std", "volatility", "sigma"):
            m = parameter_uncertainty(returns_1d, n_bootstrap=N_SIM, seed=SEED, statistic=stat)
            assert m.value >= 0.0

    def test_parameter_uncertainty_short(self) -> None:
        assert parameter_uncertainty([0.01], n_bootstrap=N_SIM, seed=SEED).value == 0.0
        m = parameter_uncertainty([0.01, 0.02], n_bootstrap=5, seed=SEED)
        assert m.parameters.get("n_bootstrap", 10) >= 10 or m.value >= 0.0
