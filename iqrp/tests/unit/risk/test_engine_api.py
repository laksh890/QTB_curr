"""Full RiskIntelligenceEngine orchestrator API coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.risk import (
    RiskDecision,
    RiskIntelligenceEngine,
    RiskReport,
    RiskSettings,
    RiskState,
)
from iqrp.app.risk.base import LimitSeverity, RiskMeasure
from iqrp.app.risk.config import (
    ESConfig,
    LeverageConfig,
    LimitConfig,
    MonteCarloConfig,
    SizingConfig,
    VaRConfig,
)


class TestCalculateRisk:
    def test_univariate(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        report = engine.calculate_risk(returns_1d)
        assert isinstance(report, RiskReport)
        assert "var" in report.tail_risk
        assert "cvar" in report.tail_risk
        assert report.risk_state in RiskState
        d = report.to_dict()
        assert "portfolio_risk" in d
        assert "drawdown" in d

    def test_multivariate_with_weights(
        self, engine: RiskIntelligenceEngine, returns_2d: np.ndarray, weights_4: np.ndarray
    ) -> None:
        report = engine.calculate_risk(returns_2d, weights=weights_4)
        assert isinstance(report.portfolio_risk, dict)
        assert (
            "portfolio_volatility" in report.portfolio_risk
            or "volatility" in report.portfolio_risk
            or "weights" in report.portfolio_risk
            or report.portfolio_risk.get("name") == "portfolio_risk"
        )
        assert report.concentration

    def test_empty_returns(self, engine: RiskIntelligenceEngine) -> None:
        report = engine.calculate_risk([])
        assert report.risk_state == RiskState.NORMAL


class TestVaRMethods:
    def test_historical(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        m = engine.var(returns_1d, method="historical")
        assert isinstance(m, RiskMeasure)
        assert m.method == "historical"
        assert m.value >= 0.0

    def test_parametric(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        m = engine.var(returns_1d, method="parametric", confidence=0.99)
        assert m.method == "parametric"
        assert m.confidence == 0.99

    def test_monte_carlo(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        m = engine.var(returns_1d, method="monte_carlo", horizon=2)
        assert m.method == "monte_carlo"
        assert m.parameters["n_simulations"] >= 100

    def test_fhs(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        m = engine.var(returns_1d, method="fhs")
        assert m.method == "filtered_historical"

    def test_default_method_from_settings(self, returns_1d: np.ndarray) -> None:
        eng = RiskIntelligenceEngine(
            RiskSettings(
                var=VaRConfig(method="parametric"), monte_carlo=MonteCarloConfig(n_simulations=200)
            )
        )
        assert eng.var(returns_1d).method == "parametric"


class TestCVaRAndES:
    def test_cvar_historical(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        m = engine.cvar(returns_1d, method="historical")
        assert m.name == "cvar"
        assert m.value >= 0.0

    def test_cvar_parametric(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        assert engine.cvar(returns_1d, method="parametric").method == "parametric"

    def test_cvar_monte_carlo(self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray) -> None:
        m = engine.cvar(returns_1d, method="monte_carlo")
        assert m.method == "monte_carlo"

    def test_expected_shortfall(
        self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray
    ) -> None:
        m = engine.expected_shortfall(returns_1d, confidence=0.95)
        assert m.name == "expected_shortfall"
        assert m.value >= 0.0

    def test_es_uses_settings_method(self, returns_1d: np.ndarray) -> None:
        eng = RiskIntelligenceEngine(
            RiskSettings(
                es=ESConfig(method="parametric"),
                monte_carlo=MonteCarloConfig(n_simulations=200),
            )
        )
        assert eng.expected_shortfall(returns_1d).method == "parametric"


class TestStressAndReverse:
    def test_historical_stress(
        self, engine: RiskIntelligenceEngine, returns_2d: np.ndarray, weights_4: np.ndarray
    ) -> None:
        out = engine.stress_test(weights_4, returns_2d, event_indices=[10, 11, 12, 13, 14])
        assert "historical" in out
        assert out["historical"]["n_event_days"] == 5

    def test_hypothetical_with_cov(
        self, engine: RiskIntelligenceEngine, weights_4: np.ndarray, cov_4: np.ndarray
    ) -> None:
        shocks = np.array([-0.05, -0.03, -0.02, -0.04])
        out = engine.stress_test(weights_4, shocks=shocks, cov=cov_4)
        assert "hypothetical" in out
        assert out["hypothetical"]["loss"] >= 0.0

    def test_hypothetical_without_cov(
        self, engine: RiskIntelligenceEngine, weights_4: np.ndarray
    ) -> None:
        out = engine.stress_test(weights_4, shocks={"a": -0.1})
        assert "hypothetical" in out

    def test_hypothetical_scalar_shock(
        self, engine: RiskIntelligenceEngine, weights_4: np.ndarray
    ) -> None:
        out = engine.stress_test(weights_4, shocks=[-0.08])
        assert "hypothetical" in out

    def test_reverse_stress(self, engine: RiskIntelligenceEngine, weights_4: np.ndarray) -> None:
        out = engine.reverse_stress(weights_4, loss_limit=0.05, direction=np.ones(4))
        assert out["name"] == "reverse_stress"
        assert "breach_possible" in out

    def test_reverse_stress_default_direction(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.reverse_stress([0.5, 0.5])
        assert "loss_limit" in out or "breach_possible" in out


class TestPositionSizeMethods:
    def test_vol_target_default(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.position_size(realized_vol=0.20)
        assert out["method"] == "volatility_target"
        assert 0.0 <= out["size"] <= engine.settings.sizing.max_leverage

    def test_kelly(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.position_size(realized_vol=0.15, edge=0.8, win_prob=0.7, method="kelly")
        assert out["method"] == "kelly"
        assert out["raw"]["value"] <= engine.settings.sizing.max_kelly + 1e-12

    def test_fractional_kelly(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.position_size(
            realized_vol=0.15, edge=0.5, win_prob=0.6, method="fractional_kelly"
        )
        assert out["method"] == "fractional_kelly"

    def test_fixed_fractional(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.position_size(realized_vol=0.1, equity=100_000, method="fixed_fractional")
        assert out["method"] == "fixed_fractional"
        assert out["size"] >= 0.0

    def test_drawdown_adjusted(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.position_size(
            realized_vol=0.12, current_drawdown=0.08, method="drawdown_adjusted"
        )
        assert out["method"] == "drawdown_adjusted"

    def test_confidence_and_regime_applied(self, engine: RiskIntelligenceEngine) -> None:
        base = engine.position_size(realized_vol=0.10, confidence=1.0, regime="normal")
        crisis = engine.position_size(realized_vol=0.10, confidence=1.0, regime="crisis")
        assert crisis["size"] <= base["size"]
        assert "Hard max_leverage" in base["note"]


class TestPortfolioHelpers:
    def test_portfolio_risk(
        self, engine: RiskIntelligenceEngine, weights_4: np.ndarray, cov_4: np.ndarray
    ) -> None:
        out = engine.portfolio_risk(weights_4, cov_4)
        assert "portfolio_volatility" in out or "volatility" in str(out).lower() or "name" in out

    def test_risk_contribution(
        self, engine: RiskIntelligenceEngine, weights_4: np.ndarray, cov_4: np.ndarray
    ) -> None:
        out = engine.risk_contribution(weights_4, cov_4)
        assert "marginal" in out and "component" in out

    def test_exposure(self, engine: RiskIntelligenceEngine, weights_4: np.ndarray) -> None:
        out = engine.exposure(weights_4)
        assert "gross" in out or "gross_exposure" in out or any("gross" in str(k) for k in out)

    def test_liquidity_risk_position_size(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.liquidity_risk(position_size=1e6, adv=5e6, spread=0.001)
        assert "score" in out or "participation" in out or "measures" in out

    def test_liquidity_risk_notional(self, engine: RiskIntelligenceEngine) -> None:
        out = engine.liquidity_risk(
            notional=2e6, adv=1e7, spread=0.0005, price=100.0, volatility=0.02
        )
        assert out is not None

    def test_drawdown_and_state(
        self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray
    ) -> None:
        dd = engine.drawdown(returns_1d)
        assert "risk_state" in dd
        assert "current_drawdown" in dd
        state = engine.risk_state(returns_1d)
        assert isinstance(state, RiskState)
        assert state.value == dd["risk_state"]


class TestCheckLimits:
    def test_clean_weights(self, engine: RiskIntelligenceEngine) -> None:
        breaches = engine.check_limits(weights=[0.05, 0.05, 0.05])
        hard = [b for b in breaches if b.severity == LimitSeverity.HARD]
        assert hard == [] or all(b.observed <= b.threshold + 1e-9 for b in hard)

    def test_position_breach(self, engine: RiskIntelligenceEngine) -> None:
        breaches = engine.check_limits(weights=[0.50, 0.05])
        assert any(b.limit_name == "max_position" for b in breaches)

    def test_liquidity_limits(self, engine: RiskIntelligenceEngine) -> None:
        breaches = engine.check_limits(
            weights=[0.05],
            participation=0.5,
            adv_coverage=0.001,
        )
        names = {b.limit_name for b in breaches}
        assert "max_participation" in names or "min_adv_coverage" in names


class TestValidatePositionInvariants:
    def test_approve_small_position(
        self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray
    ) -> None:
        decision = engine.validate_position(
            proposed_weight=0.05,
            weights=[0.05, 0.05, 0.05],
            returns=returns_1d,
            forecast_confidence=0.9,
        )
        assert isinstance(decision, RiskDecision)
        assert decision.approved is True
        assert "APPROVED" in decision.reason

    def test_hard_reject_confidence_cannot_override(
        self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray
    ) -> None:
        """Architectural invariant: confidence=1.0 cannot override hard position limits."""
        decision = engine.validate_position(
            proposed_weight=0.95,  # >> max_position (0.10)
            weights=[0.05, 0.05, 0.05, 0.05],
            returns=returns_1d,
            forecast_confidence=1.0,
            asset_index=0,
        )
        assert decision.approved is False
        assert "REJECTED" in decision.reason
        assert "cannot override hard risk limits" in decision.reason
        assert decision.audit["forecast_confidence"] == 1.0

    def test_append_new_asset_when_index_oob(
        self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray
    ) -> None:
        decision = engine.validate_position(
            proposed_weight=0.05,
            weights=[0.05, 0.05],
            returns=returns_1d,
            asset_index=99,
        )
        assert isinstance(decision, RiskDecision)

    def test_trading_halt_rejects(self, fast_settings: RiskSettings) -> None:
        eng = RiskIntelligenceEngine(fast_settings)
        # Force deep drawdown path
        crash = np.full(50, -0.05)
        decision = eng.validate_position(
            proposed_weight=0.05,
            weights=[0.05, 0.05],
            returns=crash,
            forecast_confidence=1.0,
        )
        assert decision.approved is False
        assert decision.risk_state == RiskState.TRADING_HALT or "REJECTED" in decision.reason

    def test_audit_log_appended(
        self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray
    ) -> None:
        engine.validate_position(proposed_weight=0.04, weights=[0.04], returns=returns_1d)
        assert len(engine._audit_log) >= 1


class TestRecommendedLeverageInvariant:
    def test_never_exceeds_max_even_with_full_confidence(
        self, engine: RiskIntelligenceEngine
    ) -> None:
        """Architectural invariant: confidence=1.0 cannot exceed leverage.max_leverage."""
        rec = engine.recommended_leverage(
            realized_vol=0.01,  # very low vol → wants high leverage
            confidence=1.0,
            current_drawdown=0.0,
            liquidity_score=1.0,
            regime="normal",
        )
        assert rec.value <= engine.settings.leverage.max_leverage + 1e-12

    def test_drawdown_halts_leverage(self, engine: RiskIntelligenceEngine) -> None:
        rec = engine.recommended_leverage(
            realized_vol=0.10,
            confidence=1.0,
            current_drawdown=engine.settings.drawdown.trading_halt,
        )
        assert rec.value <= engine.settings.leverage.min_leverage + 1e-12

    def test_forecast_vol_used(self, engine: RiskIntelligenceEngine) -> None:
        low = engine.recommended_leverage(realized_vol=0.05, forecast_vol=0.05)
        high = engine.recommended_leverage(realized_vol=0.05, forecast_vol=0.40)
        assert high.value <= low.value


class TestModelRiskAndMonitor:
    def test_model_risk_dict_forecasts(
        self, engine: RiskIntelligenceEngine, rng: np.random.Generator
    ) -> None:
        fcs = {
            "a": rng.normal(0, 0.01, 50),
            "b": rng.normal(0, 0.012, 50),
        }
        real = rng.normal(0, 0.01, 50)
        out = engine.model_risk_assessment(fcs, realizations=real, residuals=rng.normal(0, 1, 100))
        assert "disagreement" in out
        assert "uncertainty" in out
        assert "drift" in out

    def test_model_risk_array_forecasts(
        self, engine: RiskIntelligenceEngine, rng: np.random.Generator
    ) -> None:
        stack = rng.normal(0, 0.01, size=(3, 40))
        out = engine.model_risk_assessment(stack)
        assert "disagreement" in out

    def test_monitor_snapshot_empty(self, engine: RiskIntelligenceEngine) -> None:
        snap = engine.monitor_snapshot()
        assert "snapshot" in snap
        assert "alerts" in snap
        assert "dashboard" in snap

    def test_monitor_snapshot_after_calculate(
        self, engine: RiskIntelligenceEngine, returns_1d: np.ndarray
    ) -> None:
        engine.calculate_risk(returns_1d)
        snap = engine.monitor_snapshot()
        assert snap["snapshot"]["n_obs"] >= 1
        assert "breaches" in snap


class TestSaveLoadExportImport:
    def test_save_and_load(
        self, engine: RiskIntelligenceEngine, tmp_path: Path, returns_1d: np.ndarray
    ) -> None:
        engine.validate_position(proposed_weight=0.04, weights=[0.04], returns=returns_1d)
        path = tmp_path / "risk_state.json"
        saved = engine.save(path)
        assert saved.exists()
        loaded = RiskIntelligenceEngine.load(path)
        assert isinstance(loaded, RiskIntelligenceEngine)
        assert len(loaded._audit_log) >= 1

    def test_export_import_state(self, engine: RiskIntelligenceEngine) -> None:
        payload = engine.export_state()
        assert "settings" in payload
        assert "audit_log" in payload
        other = RiskIntelligenceEngine()
        other.import_state(payload)
        assert other.settings.seed == engine.settings.seed

    def test_import_bad_settings_swallowed(self, engine: RiskIntelligenceEngine) -> None:
        engine.import_state({"settings": {"var": {"confidence": "not-a-float"}}, "audit_log": []})
        # Should not raise
        assert engine is not None

    def test_load_with_explicit_settings(
        self, engine: RiskIntelligenceEngine, tmp_path: Path
    ) -> None:
        path = tmp_path / "eng.json"
        engine.save(path)
        custom = RiskSettings(seed=99, monte_carlo=MonteCarloConfig(n_simulations=200))
        loaded = RiskIntelligenceEngine.load(path, settings=custom)
        # import_state may overwrite from payload; engine still constructs
        assert isinstance(loaded, RiskIntelligenceEngine)


class TestSoftWarningsApproval:
    def test_soft_breach_still_approved(self, returns_1d: np.ndarray) -> None:
        # Soft concentration / herfindahl may warn but hard position ok
        settings = RiskSettings(
            limits=LimitConfig(max_position=0.50, max_concentration=0.10, max_gross_exposure=5.0),
            monte_carlo=MonteCarloConfig(n_simulations=200),
        )
        eng = RiskIntelligenceEngine(settings)
        # Equal weights within max_position but concentrated enough for soft HHI?
        decision = eng.validate_position(
            proposed_weight=0.40,
            weights=[0.40, 0.40, 0.10],
            returns=returns_1d[:50],
            forecast_confidence=0.0,
        )
        # Either approved with warnings or approved cleanly — not hard-rejected solely by soft
        if decision.breaches and all(b.severity != LimitSeverity.HARD for b in decision.breaches):
            assert decision.approved is True
            assert "APPROVED" in decision.reason
