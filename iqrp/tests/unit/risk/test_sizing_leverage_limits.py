"""Position sizing, leverage, and hard/soft limit checkers."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.risk.base import LimitSeverity, RiskLimit, evaluate_limits
from iqrp.app.risk.leverage.dynamic_leverage import recommended_leverage
from iqrp.app.risk.leverage.leverage_limits import clip_leverage
from iqrp.app.risk.limits import (
    build_concentration_limits,
    build_default_limits,
    build_exposure_limits,
    build_liquidity_limits,
    build_loss_limits,
    build_position_limits,
    check_all_limits,
    check_concentration_limits,
    check_exposure_limits,
    check_liquidity_limits,
    check_loss_limits,
    check_position_limits,
    check_positions,
)
from iqrp.app.risk.sizing.drawdown_adjusted import drawdown_adjusted_size
from iqrp.app.risk.sizing.fractional_kelly import fractional_kelly
from iqrp.app.risk.sizing.kelly import kelly_fraction
from iqrp.app.risk.sizing.risk_parity import equal_risk_contribution, risk_parity_weights
from iqrp.app.risk.sizing.volatility_target import (
    confidence_adjusted_size,
    fixed_fractional_size,
    regime_adjusted_size,
    volatility_target_size,
)


class TestKelly:
    def test_binary_capped(self) -> None:
        m = kelly_fraction(edge=0.5, win_prob=0.9, odds=1.0, max_kelly=0.5)
        assert m.value <= 0.5 + 1e-12
        assert m.parameters["capped"] is True

    def test_never_exceeds_max_kelly(self) -> None:
        """Architectural invariant: kelly fraction never exceeds max_kelly."""
        for max_k in (0.1, 0.25, 0.5, 1.0):
            m = kelly_fraction(edge=10.0, win_prob=0.99, odds=5.0, max_kelly=max_k)
            assert m.value <= max_k + 1e-12

    def test_variance_form(self) -> None:
        m = kelly_fraction(edge=0.02, variance=0.01, max_kelly=0.5)
        assert m.value == pytest.approx(min(2.0, 0.5))

    def test_edge_odds_fallback(self) -> None:
        m = kelly_fraction(edge=0.2, odds=1.0, max_kelly=0.5)
        assert m.value == pytest.approx(0.2)

    def test_negative_edge_clips_zero(self) -> None:
        m = kelly_fraction(edge=-0.5, win_prob=0.2, max_kelly=0.5)
        assert m.value == 0.0

    def test_nonfinite_raw(self) -> None:
        m = kelly_fraction(edge=float("nan"), odds=1.0, max_kelly=0.5)
        assert m.value == 0.0


class TestFractionalKelly:
    def test_scales_and_caps(self) -> None:
        full = kelly_fraction(edge=0.4, win_prob=0.7, max_kelly=0.5)
        frac = fractional_kelly(edge=0.4, win_prob=0.7, fraction=0.25, max_kelly=0.5)
        assert frac.value <= full.value + 1e-12
        assert frac.value <= 0.5 + 1e-12

    def test_fraction_clip(self) -> None:
        m = fractional_kelly(edge=0.3, win_prob=0.6, fraction=2.0, max_kelly=0.5)
        assert m.value <= 0.5 + 1e-12


class TestVolTargetAndAdjustments:
    def test_vol_target(self) -> None:
        m = volatility_target_size(realized_vol=0.20, target_vol=0.10, max_leverage=2.0)
        assert m.value == pytest.approx(0.5)

    def test_vol_target_cap(self) -> None:
        m = volatility_target_size(realized_vol=0.01, target_vol=0.10, max_leverage=2.0)
        assert m.value <= 2.0 + 1e-12

    def test_fixed_fractional(self) -> None:
        m = fixed_fractional_size(equity=100_000, risk_fraction=0.01, stop_distance=0.02)
        assert m.value == pytest.approx(50_000.0)

    def test_fixed_fractional_max_size(self) -> None:
        m = fixed_fractional_size(equity=100_000, risk_fraction=0.01, stop_distance=0.02, max_size=10_000)
        assert m.value == pytest.approx(10_000.0)

    def test_confidence_cannot_expand_beyond_one(self) -> None:
        m = confidence_adjusted_size(base_size=1.0, confidence=1.0, max_scale=1.0)
        assert m.value <= 1.0 + 1e-12
        low = confidence_adjusted_size(base_size=1.0, confidence=0.0, min_scale=0.25)
        assert low.value == pytest.approx(0.25)

    def test_regime_adjusted(self) -> None:
        base = regime_adjusted_size(base_size=1.0, regime="normal")
        crisis = regime_adjusted_size(base_size=1.0, regime="crisis")
        unknown = regime_adjusted_size(base_size=1.0, regime="weird_regime")
        assert crisis.value < base.value
        assert unknown.value == pytest.approx(0.5)

    def test_regime_custom_scales(self) -> None:
        m = regime_adjusted_size(base_size=2.0, regime="custom", regime_scales={"custom": 0.8})
        assert m.value == pytest.approx(1.6)

    def test_drawdown_adjusted(self) -> None:
        full = drawdown_adjusted_size(base_size=1.0, current_drawdown=0.0, max_drawdown_limit=0.20)
        half = drawdown_adjusted_size(base_size=1.0, current_drawdown=0.10, max_drawdown_limit=0.20)
        halt = drawdown_adjusted_size(base_size=1.0, current_drawdown=0.25, max_drawdown_limit=0.20, floor=0.0)
        assert full.value == pytest.approx(1.0)
        assert half.value == pytest.approx(0.5)
        assert halt.value == pytest.approx(0.0)


class TestRiskParity:
    def test_risk_parity_weights(self, cov_4: np.ndarray) -> None:
        out = risk_parity_weights(cov_4, max_iter=200)
        w = np.asarray(out["weights"] if "weights" in out else out.get("w", []))
        assert w.size == 4
        assert np.isclose(np.sum(w), 1.0, atol=1e-5)
        assert np.all(w >= -1e-10)

    def test_risk_parity_bad_cov(self) -> None:
        with pytest.raises(ValueError):
            risk_parity_weights(np.ones((2, 3)))

    def test_erc(self, cov_4: np.ndarray) -> None:
        out = equal_risk_contribution(cov_4)
        assert out is not None
        assert "weights" in out or "component" in out or "contributions" in out


class TestDynamicLeverage:
    def test_clip_leverage(self) -> None:
        m = clip_leverage(5.0, min_leverage=0.0, max_leverage=2.0)
        assert m.value == pytest.approx(2.0)
        assert m.parameters.get("was_clipped") is True

    def test_clip_nonfinite(self) -> None:
        assert clip_leverage(float("nan")).value == 0.0

    def test_recommended_respects_max(self) -> None:
        """Architectural invariant: recommended_leverage never exceeds max_leverage."""
        m = recommended_leverage(
            realized_vol=0.01,
            target_vol=0.10,
            confidence=1.0,
            max_leverage=2.0,
            confidence_cap=1.25,
        )
        assert m.value <= 2.0 + 1e-12

    def test_hard_drawdown_halt_ignores_confidence(self) -> None:
        m = recommended_leverage(
            realized_vol=0.05,
            current_drawdown=0.25,
            max_drawdown=0.20,
            confidence=1.0,
            min_leverage=0.0,
            max_leverage=2.0,
        )
        assert m.value == pytest.approx(0.0)
        assert m.parameters["hard_halt"] is True

    def test_regime_and_liquidity(self) -> None:
        normal = recommended_leverage(realized_vol=0.10, regime="normal", liquidity_score=1.0)
        stress = recommended_leverage(realized_vol=0.10, regime="stress", liquidity_score=0.5)
        assert stress.value <= normal.value


class TestPositionLimits:
    def test_build_and_check(self) -> None:
        lims = build_position_limits(max_position=0.10)
        assert any(L.name == "max_position" for L in lims)
        ok = check_position_limits(position_weight=0.05, max_position=0.10)
        bad = check_position_limits(position_weight=0.50, max_position=0.10)
        assert ok == []
        assert len(bad) >= 1
        assert bad[0].severity == LimitSeverity.HARD

    def test_check_positions(self) -> None:
        breaches = check_positions([0.05, 0.40, 0.05], max_position=0.10)
        assert any(b.metadata.get("index") == 1 for b in breaches)


class TestExposureLimits:
    def test_gross_and_net(self) -> None:
        lims = build_exposure_limits(max_gross_exposure=1.0, max_net_exposure=0.5)
        assert len(lims) >= 2
        breaches = check_exposure_limits([0.8, 0.8], max_gross_exposure=1.0, max_net_exposure=0.5)
        names = {b.limit_name for b in breaches}
        assert "max_gross_exposure" in names


class TestConcentrationLimits:
    def test_hard_and_soft(self) -> None:
        lims = build_concentration_limits(max_concentration=0.25, max_herfindahl=0.30)
        hhi = next(L for L in lims if L.name == "max_herfindahl")
        assert hhi.severity == LimitSeverity.SOFT
        breaches = check_concentration_limits([1.0, 0.0, 0.0], max_concentration=0.25)
        assert any(b.limit_name == "max_concentration" for b in breaches)


class TestLossLimits:
    def test_daily_and_drawdown(self) -> None:
        lims = build_loss_limits(max_daily_loss=0.03, max_drawdown=0.20, max_weekly_loss=0.07)
        assert any(L.name == "max_drawdown" and L.severity == LimitSeverity.HARD for L in lims)
        breaches = check_loss_limits(daily_loss=-0.05, current_drawdown=0.25)
        names = {b.limit_name for b in breaches}
        assert "max_daily_loss" in names
        assert "max_drawdown" in names

    def test_weekly(self) -> None:
        breaches = check_loss_limits(
            weekly_loss=0.10,
            limits=build_loss_limits(max_weekly_loss=0.05),
        )
        assert any(b.limit_name == "max_weekly_loss" for b in breaches)


class TestLiquidityLimits:
    def test_build_and_check(self) -> None:
        lims = build_liquidity_limits(max_participation=0.10, min_adv_coverage=0.01)
        assert any(L.direction == "min" for L in lims if L.name == "min_adv_coverage")
        breaches = check_liquidity_limits(
            participation=0.5,
            adv_coverage=0.001,
            time_to_liquidate=10.0,
            max_participation=0.10,
            min_adv_coverage=0.01,
            max_time_to_liquidate=5.0,
        )
        names = {b.limit_name for b in breaches}
        assert "max_participation" in names
        assert "min_adv_coverage" in names


class TestAggregateLimits:
    def test_default_limits(self) -> None:
        lims = build_default_limits()
        assert len(lims) >= 5

    def test_check_all_clean(self) -> None:
        breaches = check_all_limits(weights=[0.05, 0.05, 0.05], daily_loss=0.0, current_drawdown=0.0)
        hard = [b for b in breaches if b.severity == LimitSeverity.HARD]
        assert hard == []

    def test_check_all_with_liquidity(self) -> None:
        breaches = check_all_limits(
            weights=[0.05],
            participation=0.01,
            adv_coverage=1.0,
        )
        assert isinstance(breaches, list)

    def test_no_confidence_override_param(self) -> None:
        # check_all_limits must not accept confidence — hard limits by design
        import inspect

        sig = inspect.signature(check_all_limits)
        assert "confidence" not in sig.parameters
        assert "forecast_confidence" not in sig.parameters


class TestEvaluateLimitsBase:
    def test_evaluate_limits(self) -> None:
        limits = [
            RiskLimit(name="max_position", threshold=0.1, severity=LimitSeverity.HARD),
            RiskLimit(name="min_adv", threshold=0.01, severity=LimitSeverity.HARD, direction="min"),
        ]
        breaches = evaluate_limits(limits, {"max_position": 0.5, "min_adv": 0.001})
        assert len(breaches) == 2
        # Missing key skipped
        assert evaluate_limits(limits, {"other": 1.0}) == []
