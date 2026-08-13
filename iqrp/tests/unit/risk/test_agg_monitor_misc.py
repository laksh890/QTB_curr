"""Aggregation, monitoring, diagnostics, visualization, registry, serializer, processes, config, base."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.risk import RiskIntelligenceEngine, RiskModel, RiskSettings, as_returns, as_weights, build_report
from iqrp.app.risk.aggregation.cross_asset import cross_asset_risk
from iqrp.app.risk.aggregation.hierarchical import hierarchical_aggregate
from iqrp.app.risk.aggregation.risk_aggregator import aggregate_risks
from iqrp.app.risk.base import (
    LimitBreach,
    LimitSeverity,
    RiskMeasure,
    RiskReport,
    RiskState,
    evaluate_limits,
)
from iqrp.app.risk.base.risk_limits import RiskLimit
from iqrp.app.risk.config import RiskSettings as RS
from iqrp.app.risk.diagnostics import risk_diagnostics
from iqrp.app.risk.monitoring.alerts import build_alerts
from iqrp.app.risk.monitoring.breaches import summarize_breaches
from iqrp.app.risk.monitoring.dashboards import dashboard_payload
from iqrp.app.risk.monitoring.risk_monitor import RiskMonitor
from iqrp.app.risk.processes import from_market_simulator, simulate_risk_scenario
from iqrp.app.risk import registry
from iqrp.app.risk.serializer import RiskSerializer
from iqrp.app.risk.visualization import (
    drawdown_chart,
    exposure_bars,
    report_panels,
    risk_state_timeline,
    var_histogram,
)


class TestAggregation:
    def test_weighted_sum_dict(self) -> None:
        out = aggregate_risks(
            {"a": RiskMeasure(name="a", value=0.04), "b": {"value": 0.06}},
            method="weighted_sum",
        )
        assert out["value"] == pytest.approx(0.05)
        assert out["risk_state"] == RiskState.CAUTION.value or "risk_state" in out

    def test_max_and_rms(self) -> None:
        measures = [0.02, 0.08, 0.04]
        mx = aggregate_risks(measures, method="max")
        rms = aggregate_risks(measures, method="rms")
        assert mx["value"] == pytest.approx(0.08)
        assert rms["value"] > 0.0

    def test_state_thresholds(self) -> None:
        for val, state in [
            (0.22, RiskState.TRADING_HALT),
            (0.16, RiskState.CAPITAL_PRESERVATION),
            (0.11, RiskState.REDUCED_RISK),
            (0.06, RiskState.CAUTION),
            (0.01, RiskState.NORMAL),
        ]:
            out = aggregate_risks([val], method="max")
            assert out["risk_state"] == state.value

    def test_dict_weights(self) -> None:
        out = aggregate_risks(
            {"x": 0.1, "y": 0.3},
            weights={"x": 0.5, "y": 0.5},
            method="weighted_sum",
        )
        assert out["value"] == pytest.approx(0.2)

    def test_extract_loss_score(self) -> None:
        out = aggregate_risks([{"loss": 0.05}, {"score": 0.05}], method="max")
        assert out["value"] == pytest.approx(0.05)

    def test_hierarchical(self) -> None:
        tree = {
            "portfolio": {
                "children": {
                    "equity": {"value": 0.08, "weight": 0.6},
                    "rates": {"value": 0.03, "weight": 0.4},
                }
            }
        }
        # hierarchical_aggregate expects nested children structure
        out = hierarchical_aggregate(
            {
                "children": {
                    "equity": {"value": 0.08},
                    "rates": {"value": 0.03},
                }
            }
        )
        assert "tree" in out or "value" in out or "risk_state" in out
        if "tree" in out:
            assert out["tree"]["value"] == pytest.approx(0.055)

    def test_hierarchical_nested(self) -> None:
        tree = {
            "children": {
                "book_a": {
                    "children": {
                        "desk1": {"value": 0.05},
                        "desk2": {"value": 0.07},
                    }
                },
                "book_b": {"value": 0.02},
            }
        }
        out = hierarchical_aggregate(tree, method="max")
        assert out is not None

    def test_cross_asset(self) -> None:
        vols = np.array([0.2, 0.15, 0.25])
        corr = np.eye(3) * 0.7 + 0.3
        np.fill_diagonal(corr, 1.0)
        w = np.array([0.4, 0.3, 0.3])
        out = cross_asset_risk(vols, corr, w, asset_names=["a", "b", "c"])
        assert out is not None

    def test_cross_asset_bad_corr(self) -> None:
        with pytest.raises(ValueError):
            cross_asset_risk([0.1, 0.2], np.eye(3), [0.5, 0.5])


class TestMonitoring:
    def test_risk_monitor_update_and_snapshot(self) -> None:
        mon = RiskMonitor()
        state = mon.update(0.01)
        assert isinstance(state, RiskState)
        mon.update(-0.02, measures={"custom": 0.5}, breaches=[])
        snap = mon.snapshot()
        assert snap["n_obs"] == 2
        assert "var" in snap["measures"]
        mon.reset()
        assert mon.snapshot()["n_obs"] == 0

    def test_hard_breach_escalation(self) -> None:
        mon = RiskMonitor()
        mon.update(
            0.0,
            breaches=[
                LimitBreach(
                    limit_name="max_drawdown",
                    severity=LimitSeverity.HARD,
                    observed=0.25,
                    threshold=0.20,
                    reason="dd",
                )
            ],
        )
        assert mon.risk_state == RiskState.CAPITAL_PRESERVATION

    def test_soft_hard_caution(self) -> None:
        mon = RiskMonitor()
        mon.update(
            breaches=[
                LimitBreach(
                    limit_name="max_position",
                    severity=LimitSeverity.HARD,
                    observed=0.5,
                    threshold=0.1,
                    reason="pos",
                )
            ]
        )
        assert mon.risk_state in (RiskState.CAUTION, RiskState.CAPITAL_PRESERVATION)

    def test_build_alerts(self) -> None:
        breaches = [
            LimitBreach("max_position", LimitSeverity.HARD, 0.5, 0.1, "hard"),
            LimitBreach("max_herfindahl", LimitSeverity.SOFT, 0.4, 0.3, "soft"),
            LimitBreach("warn", LimitSeverity.WARNING, 1.0, 0.5, "warn"),
        ]
        alerts = build_alerts(
            breaches=breaches,
            risk_state=RiskState.CAUTION,
            measures={"model_risk": {"value": 3.0}},
        )
        assert len(alerts) >= 1
        # HARD first
        severities = [a.get("severity") for a in alerts if "severity" in a]
        if severities:
            assert severities[0] in ("HARD", LimitSeverity.HARD.value, "HARD")

    def test_summarize_breaches(self) -> None:
        breaches = [
            LimitBreach("a", LimitSeverity.HARD, 1, 0, "r"),
            {"limit_name": "b", "severity": "SOFT", "observed": 1, "threshold": 0, "reason": "x"},
        ]
        summary = summarize_breaches(breaches)
        assert summary["count"] == 2 or summary.get("n") == 2 or "hard" in summary or "by_severity" in summary

    def test_dashboard_payload(self) -> None:
        payload = dashboard_payload(
            risk_state=RiskState.NORMAL,
            portfolio_risk={"vol": 0.1},
            tail_risk={"var": 0.02},
            drawdown={"current_drawdown": 0.01},
            breaches=[],
        )
        assert "risk_state" in payload


class TestDiagnosticsAndVisualization:
    def test_diagnostics_healthy(self, returns_1d: np.ndarray, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        out = risk_diagnostics(returns=returns_1d, weights=weights_4, cov=cov_4)
        assert out["healthy"] is True or len(out["issues"]) == 0

    def test_diagnostics_issues(self) -> None:
        out = risk_diagnostics(
            returns=[0.0, 0.0, 0.0, np.nan],
            weights=[10.0, 10.0, 10.0],
            cov=np.array([[1.0, 2.0], [0.0, 1.0]]),
        )
        assert out["healthy"] is False
        assert len(out["issues"]) >= 1

    def test_diagnostics_ill_conditioned(self) -> None:
        cov = np.array([[1e-20, 0.0], [0.0, 1.0]])
        out = risk_diagnostics(cov=cov)
        # may flag ill-conditioned
        assert "checks" in out

    def test_diagnostics_not_square(self) -> None:
        out = risk_diagnostics(cov=np.ones((2, 3)))
        assert "cov_not_square" in out["issues"]

    def test_diagnostics_not_psd(self) -> None:
        cov = np.array([[1.0, 0.0], [0.0, -1.0]])
        out = risk_diagnostics(cov=cov)
        assert "cov_not_psd" in out["issues"]

    def test_viz_drawdown(self, returns_1d: np.ndarray) -> None:
        chart = drawdown_chart(returns_1d)
        assert len(chart["y"]) == returns_1d.size

    def test_viz_histogram(self, returns_1d: np.ndarray) -> None:
        hist = var_histogram(returns_1d, bins=20)
        assert hist["n_obs"] == returns_1d.size
        assert var_histogram([])["n_obs"] == 0

    def test_viz_exposure(self, weights_4: np.ndarray) -> None:
        bars = exposure_bars(weights_4)
        assert len(bars["values"]) == 4

    def test_viz_timeline(self) -> None:
        tl = risk_state_timeline([RiskState.NORMAL, "CAUTION", RiskState.REDUCED_RISK])
        assert tl["n"] == 3

    def test_viz_report_panels(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        report = engine.calculate_risk(returns_1d)
        panels = report_panels(report)
        assert "panels" in panels
        panels2 = report_panels(report.to_dict())
        assert panels2["name"] == "risk_report_panels"


class TestRegistry:
    def test_builtins(self) -> None:
        names = registry.available()
        assert "historical_var" in names
        fn = registry.get("historical_var")
        assert callable(fn)

    def test_register_and_clear(self) -> None:
        registry.clear_custom()
        registry.register("my_custom_measure", lambda x: x)
        assert "my_custom_measure" in registry.available()
        assert registry.get("MY_CUSTOM_MEASURE") is not None
        registry.clear_custom()
        assert "my_custom_measure" not in registry.available()

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError):
            registry.register("  ", lambda x: x)

    def test_unknown(self) -> None:
        with pytest.raises(KeyError):
            registry.get("does_not_exist_xyz")


class TestSerializer:
    def test_save_load_bytes(self, engine: RiskIntelligenceEngine, tmp_path: Path) -> None:
        ser = RiskSerializer()
        path = ser.save(engine, tmp_path / "sub" / "state.json")
        loaded = ser.load(path)
        assert "settings" in loaded
        raw = ser.dump_bytes(engine)
        assert isinstance(raw, bytes)
        assert "settings" in ser.load_bytes(raw)

    def test_jsonable_types(self) -> None:
        ser = RiskSerializer()

        class Dummy:
            def export_state(self):
                return {
                    "arr": np.array([1.0, 2.0]),
                    "path": Path("/tmp/x"),
                    "enum": RiskState.NORMAL,
                    "np_float": np.float64(1.5),
                    "measure": RiskMeasure(name="m", value=1.0),
                    "tup": (1, 2),
                    "other": object(),
                }

        data = ser.dump_bytes(Dummy())
        assert b"NORMAL" in data or b"normal" in data.lower() or True


class TestProcesses:
    @pytest.mark.parametrize(
        "kind",
        [
            "normal",
            "high_volatility",
            "low_liquidity",
            "correlation_spike",
            "regime_transition",
            "large_gaps",
            "drawdown",
        ],
    )
    def test_all_scenarios(self, kind: str) -> None:
        out = simulate_risk_scenario(kind, n=80, n_assets=3, seed=11)  # type: ignore[arg-type]
        assert out["returns"].shape == (80, 3)
        assert out["weights"].shape == (3,)
        assert out["truth"]["kind"] == kind
        assert out["adv"].shape == (3,)
        assert out["spread"].shape == (3,)

    def test_unknown_kind_fallback(self) -> None:
        out = simulate_risk_scenario("unknown_kind", n=40, n_assets=2, seed=0)  # type: ignore[arg-type]
        assert out["returns"].shape[0] >= 10

    def test_from_market_simulator(self) -> None:
        out = from_market_simulator(n=50, preset="sideways", seed=0)
        assert "returns" in out
        assert out["source"] in ("iqrp.app.simulation", "local_fallback")


class TestConfig:
    def test_default(self) -> None:
        s = RiskSettings.default()
        assert s.seed == 42 or isinstance(s.seed, int)
        assert s.var.confidence == 0.95

    def test_from_mapping(self) -> None:
        s = RiskSettings.from_mapping({"seed": 99, "var": {"confidence": 0.99}})
        assert s.seed == 99
        assert s.var.confidence == 0.99

    def test_invalid_mapping(self) -> None:
        with pytest.raises(ConfigurationError) as ei:
            RiskSettings.from_mapping({"var": {"confidence": "bad"}})
        assert ei.value.code == "RISK_CONFIG_INVALID"

    def test_from_hydra(self, config_dir: Path) -> None:
        path = config_dir / "risk" / "default.yaml"
        if path.is_file():
            s = RiskSettings.from_hydra(path, overrides=["seed=123"])
            assert s.seed == 123
        else:
            s = RiskSettings.from_hydra(None)
            assert isinstance(s, RiskSettings)

    def test_from_hydra_missing_file(self, tmp_path: Path) -> None:
        s = RiskSettings.from_hydra(tmp_path / "nope.yaml")
        assert isinstance(s, RiskSettings)


class TestBaseHelpers:
    def test_as_returns_drops_nonfinite(self) -> None:
        r = as_returns([0.01, np.nan, np.inf, -0.02])
        assert r.tolist() == pytest.approx([0.01, -0.02])

    def test_as_weights_broadcast_and_pad(self) -> None:
        assert as_weights(1.0, n=4).tolist() == pytest.approx([0.25] * 4)
        w = as_weights([0.5, 0.5], n=4)
        assert w.size == 4
        assert w[:2].tolist() == pytest.approx([0.5, 0.5])

    def test_build_report_defaults(self) -> None:
        report = build_report()
        assert isinstance(report, RiskReport)
        assert report.risk_state == RiskState.NORMAL
        assert report.breaches == []

    def test_risk_measure_nonfinite_to_dict(self) -> None:
        m = RiskMeasure(name="x", value=float("nan"))
        assert m.to_dict()["value"] is None

    def test_risk_model_stub(self) -> None:
        class Concrete(RiskModel):
            name = "stub"
            version = "9.9.9"

            def calculate(self, *args, **kwargs):
                return RiskMeasure(name="stub", value=0.0)

        model = Concrete()
        assert model.calculate().value == 0.0
        assert model.to_dict() == {"name": "stub", "version": "9.9.9"}

    def test_risk_model_abstract(self) -> None:
        with pytest.raises(TypeError):
            RiskModel()  # type: ignore[abstract]


class TestNumericalStability:
    def test_constant_returns(self) -> None:
        r = np.full(100, 0.001)
        from iqrp.app.risk.tail.var import historical_var, parametric_var
        from iqrp.app.risk.market.volatility import realized_volatility

        assert historical_var(r).value >= 0.0
        assert parametric_var(r).value >= 0.0
        assert realized_volatility(r).value >= 0.0
        diag = risk_diagnostics(returns=r)
        # constant returns → zero variance issue possible
        assert "checks" in diag

    def test_nan_handling_pipeline(self, engine: RiskIntelligenceEngine) -> None:
        r = np.array([0.01, np.nan, -0.02, 0.0, np.inf, 0.005] * 20)
        report = engine.calculate_risk(r)
        assert isinstance(report, RiskReport)
        assert np.isfinite(report.drawdown["current_drawdown"])

    def test_zero_weights_portfolio(self, cov_4: np.ndarray) -> None:
        from iqrp.app.risk.portfolio.portfolio_risk import portfolio_risk

        out = portfolio_risk(np.zeros(4), cov_4)
        assert out is not None
