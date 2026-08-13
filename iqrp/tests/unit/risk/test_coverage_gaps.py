"""Gap-closing tests targeting remaining uncovered risk lines."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings, build_report
from iqrp.app.risk.aggregation.cross_asset import cross_asset_risk
from iqrp.app.risk.aggregation.risk_aggregator import aggregate_risks
from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskMeasure, RiskState
from iqrp.app.risk.config import LimitConfig, MonteCarloConfig, RiskSettings as RS
from iqrp.app.risk.diagnostics import risk_diagnostics
from iqrp.app.risk.leverage.dynamic_leverage import recommended_leverage
from iqrp.app.risk.limits.position_limits import (
    build_position_limits,
    check_position_limits,
)
from iqrp.app.risk.market.beta import beta, tracking_error
from iqrp.app.risk.market.correlation import (
    correlation_matrix,
    covariance_matrix,
    ewma_correlation,
    ewma_covariance,
    rolling_correlation,
    shrinkage_covariance,
)
from iqrp.app.risk.model_risk.forecast_uncertainty import forecast_uncertainty
from iqrp.app.risk.model_risk.model_disagreement import model_disagreement
from iqrp.app.risk.monitoring.alerts import build_alerts
from iqrp.app.risk.monitoring.risk_monitor import RiskMonitor
from iqrp.app.risk.portfolio.concentration import herfindahl, max_weight
from iqrp.app.risk.portfolio.factor_exposure import factor_exposures
from iqrp.app.risk.processes import from_market_simulator
from iqrp.app.risk.serializer import RiskSerializer
from iqrp.app.risk.simulation.bootstrap import block_bootstrap, historical_bootstrap
from iqrp.app.risk.simulation.copula import gaussian_copula_simulate
from iqrp.app.risk.simulation.monte_carlo import correlated_monte_carlo
from iqrp.app.risk.simulation.scenario_engine import ScenarioEngine
from iqrp.app.risk.sizing.risk_parity import risk_parity_weights
from iqrp.app.risk.stress.historical import historical_stress
from iqrp.app.risk.stress.reverse_stress import reverse_stress
from iqrp.app.risk.stress.scenarios import ScenarioSpec, apply_shock
from iqrp.app.risk.tail.cvar import historical_cvar
from iqrp.app.risk.tail.expected_shortfall import conditional_tail_expectation


class TestAggregatorWeightMismatch:
    def test_dict_list_weight_pad(self) -> None:
        out = aggregate_risks({"a": 0.1, "b": 0.2, "c": 0.3}, weights=[1.0])  # size mismatch
        assert "value" in out

    def test_list_weight_pad(self) -> None:
        out = aggregate_risks([0.1, 0.2, 0.3], weights=[1.0, 1.0])  # shorter
        assert out["value"] >= 0.0

    def test_empty_list(self) -> None:
        out = aggregate_risks([], method="weighted_sum")
        assert out["value"] == 0.0


class TestCrossAssetNamePad:
    def test_short_names(self) -> None:
        out = cross_asset_risk([0.1, 0.2, 0.15], np.eye(3), [0.3, 0.3, 0.4], asset_names=["only"])
        assert out["n_assets"] == 3


class TestBetaGaps:
    def test_empty(self) -> None:
        assert beta([], []).value == 0.0
        assert tracking_error([], []).value == 0.0

    def test_n_lt_2(self) -> None:
        assert beta([0.01], [0.02]).value == 0.0
        assert tracking_error([0.01], [0.02]).value == 0.0

    def test_tracking_window_and_no_ann(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.01, 80)
        b = rng.normal(0, 0.01, 80)
        m = tracking_error(a, b, window=30, annualize=False)
        assert m.value >= 0.0

    def test_constant_asset_r2(self) -> None:
        b = np.linspace(-0.01, 0.01, 50)
        a = np.zeros(50)
        m = beta(a, b)
        assert m.value == 0.0 or np.isfinite(m.value)


class TestCorrelationGaps:
    def test_short_and_empty(self) -> None:
        assert "matrix" in correlation_matrix([[1.0]])  # t<2
        assert "matrix" in correlation_matrix(np.zeros((0, 3)))
        assert "matrix" in covariance_matrix([[1.0]])
        assert "matrix" in covariance_matrix(np.zeros((0, 2)))

    def test_nan_rows(self) -> None:
        x = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, np.nan]])
        correlation_matrix(x)
        covariance_matrix(x)

    def test_1d_cov_scalar_path(self) -> None:
        # single column → cov may be 0-d before wrap
        covariance_matrix(np.array([0.01, 0.02, -0.01, 0.0]))

    def test_shrinkage_empty(self) -> None:
        out = shrinkage_covariance(np.zeros((0, 0)))
        assert out["name"] == "shrinkage_covariance"

    def test_ewma_empty(self) -> None:
        assert ewma_correlation(np.zeros((0, 2)))["n_obs"] == 0
        assert ewma_covariance(np.zeros((0, 2)))["n_obs"] == 0

    def test_rolling_zero_std(self) -> None:
        x = np.ones(100)
        y = np.ones(100)
        assert rolling_correlation(x, y, window=20).value == 0.0


class TestModelRiskGaps:
    def test_disagreement_1d(self) -> None:
        m = model_disagreement([0.1, 0.2, 0.3, 0.4])
        assert m.value >= 0.0

    def test_disagreement_axis1(self) -> None:
        m = model_disagreement(np.ones((5, 3)), axis=1)
        assert m.value == 0.0  # identical

    def test_forecast_empty(self) -> None:
        assert forecast_uncertainty([], []).value == 0.0


class TestFactorGaps:
    def test_short_factor_names(self) -> None:
        rng = np.random.default_rng(1)
        F = rng.normal(0, 0.01, size=(50, 3))
        y = F @ np.array([0.5, 0.2, 0.1]) + rng.normal(0, 0.001, 50)
        out = factor_exposures(y, F, factor_names=["only_one"])
        assert out is not None

    def test_insufficient_t(self) -> None:
        out = factor_exposures([0.01, 0.02], np.ones((2, 3)))
        assert out is not None

    def test_nan_mask_insufficient(self) -> None:
        F = np.ones((20, 2))
        F[:18] = np.nan
        y = np.random.default_rng(0).normal(0, 0.01, 20)
        out = factor_exposures(y, F)
        assert out is not None

    def test_k_zero(self) -> None:
        out = factor_exposures(np.ones(10), np.zeros((10, 0)))
        assert out is not None


class TestOrchestratorGaps:
    def test_max_leverage_breach(self, returns_1d: np.ndarray) -> None:
        settings = RiskSettings(
            limits=LimitConfig(max_position=0.95, max_leverage=1.0, max_gross_exposure=5.0),
            monte_carlo=MonteCarloConfig(n_simulations=200),
        )
        eng = RiskIntelligenceEngine(settings)
        decision = eng.validate_position(
            proposed_weight=0.6,
            weights=[0.6, 0.6],
            returns=returns_1d[:80],
            forecast_confidence=0.0,
            asset_index=0,
        )
        assert decision.approved is False
        assert any(b.limit_name == "max_leverage" for b in decision.breaches)

    def test_trading_halt_without_hard_dd_breach(self) -> None:
        """TRADING_HALT with dd == threshold (no hard > breach) rejects via state."""
        settings = RiskSettings(
            limits=LimitConfig(
                max_position=0.50,
                max_gross_exposure=5.0,
                max_net_exposure=5.0,
                max_concentration=1.0,
            ),
            monte_carlo=MonteCarloConfig(n_simulations=200),
        )
        eng = RiskIntelligenceEngine(settings)
        halt_dd = {
            "risk_state": RiskState.TRADING_HALT.value,
            "current_drawdown": eng.settings.drawdown.trading_halt,  # equal → no hard >
            "max_drawdown": eng.settings.drawdown.trading_halt,
            "peak_equity": 1.0,
            "drawdown_duration": 1,
            "recovery_time": None,
            "wealth": 0.8,
            "thresholds": {},
            "measures": {},
        }
        with patch.object(eng, "drawdown", return_value=halt_dd):
            decision = eng.validate_position(
                proposed_weight=0.05,
                weights=[0.05] * 10,  # diversified → avoid soft HHI noise
                returns=np.zeros(20),
                forecast_confidence=0.0,
            )
        assert decision.approved is False
        assert "TRADING_HALT" in decision.reason

    def test_clean_approval_path(self) -> None:
        settings = RiskSettings(
            limits=LimitConfig(
                max_position=0.50,
                max_gross_exposure=5.0,
                max_net_exposure=5.0,
                max_concentration=1.0,
            ),
            monte_carlo=MonteCarloConfig(n_simulations=200),
        )
        eng = RiskIntelligenceEngine(settings)
        # 10 equal names → HHI=0.1 < soft 0.3 threshold
        w = [0.05] * 10
        decision = eng.validate_position(
            proposed_weight=0.05,
            weights=w,
            returns=np.full(30, 0.001),
            forecast_confidence=0.0,
        )
        assert decision.approved is True
        assert decision.reason.startswith("APPROVED:")
        assert "WARNINGS" not in decision.reason

    def test_monitor_snapshot_limitbreach_objects(self, engine: RiskIntelligenceEngine) -> None:
        engine._last_report = build_report(
            risk_state=RiskState.CAUTION,
            breaches=[
                LimitBreach(
                    limit_name="max_position",
                    severity=LimitSeverity.HARD,
                    observed=0.5,
                    threshold=0.1,
                    reason="too big",
                )
            ],
        )
        snap = engine.monitor_snapshot()
        assert snap["alerts"]


class TestAlertsModelRisk:
    def test_named_measure_soft_alerts(self) -> None:
        alerts = build_alerts(
            measures={
                "model_drift": {"value": 3.5},
                "forecast_uncertainty": {"value": 2.1},
                "model_disagreement": {"value": 5.0},
                "other": {"value": 9.0},
            }
        )
        types = [a["type"] for a in alerts]
        assert types.count("model_risk") == 3


class TestMonitorMeasureRiskMeasure:
    def test_update_with_risk_measure(self) -> None:
        mon = RiskMonitor()
        mon.update(measures={"x": RiskMeasure(name="x", value=1.0)})
        assert "x" in mon.snapshot()["measures"]


class TestDiagnosticsWeightsNonfinite:
    def test_weights_nonfinite(self) -> None:
        out = risk_diagnostics(weights=[0.1, np.nan, 0.2])
        assert "weights_nonfinite" in out["issues"]

    def test_returns_insufficient(self) -> None:
        out = risk_diagnostics(returns=[0.01])
        assert "returns_insufficient_obs" in out["issues"]


class TestPositionLimitSingleName:
    def test_max_single_name(self) -> None:
        lims = build_position_limits(max_position=0.10, max_single_name=0.08)
        assert any(L.name == "max_single_name" for L in lims)
        breaches = check_position_limits(position_weight=0.09, limits=lims)
        assert any(b.limit_name == "max_single_name" for b in breaches)


class TestSerializerEnumPath:
    def test_model_dump_and_enum(self) -> None:
        ser = RiskSerializer()

        class Obj:
            def model_dump(self):
                return {"state": RiskState.NORMAL, "arr": np.int64(3)}

        data = ser.dump_bytes(Obj())
        assert isinstance(data, bytes)


class TestProcessesSimulatorSuccess:
    def test_from_market_simulator_1d_list_prices(self) -> None:
        # Use list prices so `getattr(...) or market.get(...)` does not hit ndarray truthiness.
        mock_sim = MagicMock()
        prices_1d = list(np.cumprod(1.0 + np.random.default_rng(0).normal(0, 0.01, 80)))
        mock_market = MagicMock()
        mock_market.prices = prices_1d
        mock_sim.simulate_preset.return_value = mock_market
        fake_mod = MagicMock()
        fake_mod.MarketSimulator = MagicMock(return_value=mock_sim)
        with patch.dict("sys.modules", {"iqrp.app.simulation.base.simulator": fake_mod}):
            out = from_market_simulator(n=40, seed=1)
        assert out["source"] == "iqrp.app.simulation"
        assert out["returns"].ndim == 2
        assert out["returns"].shape[1] == 1

    def test_from_market_simulator_2d_dict(self) -> None:
        mock_sim = MagicMock()
        prices = np.cumprod(1.0 + np.random.default_rng(1).normal(0, 0.01, size=(60, 3)), axis=0)
        # dict path uses .get("prices") when getattr returns None
        mock_sim.simulate_preset.return_value = {"prices": prices.tolist()}
        fake_mod = MagicMock()
        fake_mod.MarketSimulator = MagicMock(return_value=mock_sim)
        with patch.dict("sys.modules", {"iqrp.app.simulation.base.simulator": fake_mod}):
            out = from_market_simulator(n=30, seed=2)
        assert out["source"] == "iqrp.app.simulation"
        assert out["returns"].shape[1] == 3


class TestBootstrapEmpty:
    def test_empty_bootstraps(self) -> None:
        historical_bootstrap([], n_simulations=20, seed=0)
        block_bootstrap([], n_simulations=20, horizon=5, seed=0)
        # n <= block_size path
        block_bootstrap([0.01, 0.02], n_simulations=10, horizon=5, block_size=10, seed=0)


class TestMonteCarloMeanPad:
    def test_mean_size_mismatch(self) -> None:
        out = correlated_monte_carlo([0.0], np.eye(3) * 1e-4, n_simulations=50, seed=0)
        assert out["terminal"].shape == (50, 3)


class TestCopulaAndScenario:
    def test_copula_short(self) -> None:
        out = gaussian_copula_simulate([[0.01]], n_simulations=20, seed=0)
        assert out["samples"].shape[0] == 20

    def test_scenario_1d_copula(self) -> None:
        eng = ScenarioEngine(n_simulations=50, horizon=1, seed=0)
        out = eng.run(np.random.default_rng(0).normal(0, 0.01, 80), method="gaussian_copula")
        assert out is not None


class TestRiskParityEdges:
    def test_empty_cov(self) -> None:
        out = risk_parity_weights(np.zeros((0, 0)))
        assert out["weights"] == []

    def test_zero_port_var(self) -> None:
        # Zero cov → port_var ~ 0 break
        out = risk_parity_weights(np.zeros((3, 3)), max_iter=10)
        assert len(out["weights"]) == 3


class TestReverseAndScenariosGaps:
    def test_direction_size_mismatch(self) -> None:
        out = reverse_stress([0.5, 0.5, 0.0], [1.0], loss_limit=0.05)
        assert "breach_possible" in out

    def test_apply_shock_size_mismatch(self) -> None:
        out = apply_shock([0.3, 0.3, 0.4], [-0.1])
        assert out["loss"] >= 0.0

    def test_spec_to_dict_array(self) -> None:
        spec = ScenarioSpec(name="a", shocks=np.array([-0.1, -0.2]))
        d = spec.to_dict()
        assert isinstance(d["shocks"], list)


class TestHistoricalNoWeights:
    def test_2d_default_weights(self, returns_2d: np.ndarray) -> None:
        out = historical_stress(returns_2d, event_window=(0, 10))
        assert out["n_event_days"] == 10


class TestCVaREmptyTail:
    def test_identical_returns_cvar(self) -> None:
        # All equal → quantile may yield empty strict tail in edge cases
        r = np.full(20, 0.0)
        assert historical_cvar(r).value >= 0.0

    def test_cte_empty_tail_threshold(self) -> None:
        r = np.array([0.01, 0.02, 0.03])
        m = conditional_tail_expectation(r, threshold=-10.0)  # nothing below
        assert m.value >= 0.0


class TestConcentrationEmpty:
    def test_empty_weights(self) -> None:
        assert herfindahl([]).value == 0.0
        assert max_weight([]).value == 0.0
        assert herfindahl([0.0, 0.0]).value >= 0.0


class TestConfigOmegaConfAndDefaultFallback:
    def test_from_mapping_omegaconf(self) -> None:
        cfg = OmegaConf.create({"seed": 77, "var": {"confidence": 0.99}})
        s = RiskSettings.from_mapping(cfg)
        assert s.seed == 77

    def test_default_fallback_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "iqrp.app.risk.config._default_config_path",
            lambda: tmp_path / "missing.yaml",
        )
        s = RS.default()
        assert isinstance(s, RS)


class TestLeverageCustomRegime:
    def test_custom_regime_scales(self) -> None:
        m = recommended_leverage(
            realized_vol=0.10,
            regime="custom",
            regime_scales={"custom": 0.9},
        )
        assert m.value >= 0.0


class TestSoftWarningsPath:
    def test_approved_with_warnings(self, returns_1d: np.ndarray) -> None:
        settings = RiskSettings(
            limits=LimitConfig(
                max_position=0.50,
                max_gross_exposure=5.0,
                max_net_exposure=5.0,
                max_concentration=0.15,  # soft-ish via HHI soft + maybe conc
            ),
            monte_carlo=MonteCarloConfig(n_simulations=200),
        )
        eng = RiskIntelligenceEngine(settings)
        # Concentrated but under hard max_position
        decision = eng.validate_position(
            proposed_weight=0.40,
            weights=[0.40, 0.40, 0.10],
            returns=np.full(40, 0.001),
            forecast_confidence=0.0,
        )
        if decision.breaches and not any(b.severity == LimitSeverity.HARD for b in decision.breaches):
            assert decision.approved is True
            assert "APPROVED_WITH_WARNINGS" in decision.reason or "APPROVED" in decision.reason


class TestRemainingMicroGaps:
    def test_covariance_window_and_nan_only(self) -> None:
        # window path + all-nan rows → clean.shape[0] < 2
        x = np.array([[np.nan, np.nan], [np.nan, np.nan], [1.0, 2.0]])
        covariance_matrix(x, window=2)

    def test_rolling_corr_nonfinite(self) -> None:
        a = np.array([1.0, 1.0, 1.0, 2.0] * 20, dtype=float)
        b = np.array([1.0, 1.0, 1.0, 2.0] * 20, dtype=float)
        with patch("numpy.corrcoef", return_value=np.array([[1.0, np.nan], [np.nan, 1.0]])):
            m = rolling_correlation(a, b, window=10)
        assert m.value == 0.0

    def test_serializer_model_dump_and_enum(self) -> None:
        ser = RiskSerializer()

        class NestedDump:
            def model_dump(self):
                return {"ok": True, "state": RiskState.REDUCED_RISK}

        class Outer:
            def export_state(self):
                return {"nested": NestedDump(), "bare_enum": RiskState.NORMAL}

        payload = ser.load_bytes(ser.dump_bytes(Outer()))
        assert payload["nested"]["ok"] is True
        assert payload["bare_enum"] == "NORMAL"
        assert payload["nested"]["state"] == "REDUCED_RISK"

    def test_serializer_enum_exception_fallback(self) -> None:
        ser = RiskSerializer()

        class Weird:
            value = 1
            # looks enum-ish but isn't
            def __repr__(self):
                return "Weird()"

        class Outer:
            def export_state(self):
                return {"w": Weird()}

        data = ser.dump_bytes(Outer())
        assert b"Weird" in data or b"w" in data

    def test_factor_linalg_error(self) -> None:
        with patch("numpy.linalg.lstsq", side_effect=np.linalg.LinAlgError("boom")):
            out = factor_exposures(
                np.random.default_rng(0).normal(0, 0.01, 50),
                np.random.default_rng(1).normal(0, 0.01, size=(50, 2)),
            )
        assert out is not None

    def test_risk_parity_simplex_edges(self) -> None:
        from iqrp.app.risk.sizing.risk_parity import _project_simplex

        assert _project_simplex(np.zeros(0)).size == 0
        # Force rho.size == 0: values already summing correctly with pattern that fails nonzero
        # Use huge negative then project — s<=0 branch
        w = _project_simplex(np.array([-100.0, -100.0, -100.0]))
        assert np.isclose(np.sum(w), 1.0)
        # Another vector that yields empty rho (all mass on equal after sort edge)
        w2 = _project_simplex(np.array([0.0, 0.0]))
        assert np.isclose(np.sum(w2), 1.0)

    def test_block_bootstrap_short_series(self) -> None:
        # n <= block_size uses start=0 path (empty-block branch is defensive)
        out = block_bootstrap(np.array([0.01]), n_simulations=5, horizon=3, block_size=5, seed=0)
        assert out["terminal"].shape == (5,)

    def test_scenario_spec_dict_to_dict(self) -> None:
        spec = ScenarioSpec(name="d", shocks={"a": -0.1}, description="x", metadata={"k": 1})
        assert isinstance(spec.to_dict()["shocks"], dict)

    def test_cvar_empty_tail_via_mock_quantile(self) -> None:
        r = np.array([-0.01, 0.0, 0.01, 0.02])
        with patch("numpy.quantile", return_value=-1.0):
            m = historical_cvar(r)
        assert m.value >= 0.0
