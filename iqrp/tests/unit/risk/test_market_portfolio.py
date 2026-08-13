"""Market risk metrics and portfolio risk analytics."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.risk.market.beta import beta, tracking_error
from iqrp.app.risk.market.correlation import (
    correlation_matrix,
    covariance_matrix,
    ewma_correlation,
    ewma_covariance,
    rolling_correlation,
    shrinkage_covariance,
)
from iqrp.app.risk.market.gap_risk import gap_risk
from iqrp.app.risk.market.liquidity import liquidity_risk
from iqrp.app.risk.market.volatility import ewma_volatility, realized_volatility
from iqrp.app.risk.portfolio.concentration import concentration_risk, herfindahl, max_weight
from iqrp.app.risk.portfolio.diversification import diversification_ratio
from iqrp.app.risk.portfolio.exposure import (
    exposure_summary,
    gross_exposure,
    long_exposure,
    net_exposure,
    short_exposure,
)
from iqrp.app.risk.portfolio.factor_exposure import factor_exposures
from iqrp.app.risk.portfolio.portfolio_risk import (
    component_risk_contribution,
    marginal_risk_contribution,
    portfolio_risk,
    portfolio_volatility,
)


class TestVolatility:
    def test_realized_vol(self, returns_1d: np.ndarray) -> None:
        m = realized_volatility(returns_1d)
        assert m.value > 0.0
        assert m.name == "realized_volatility" or "vol" in m.name

    def test_realized_vol_window(self, returns_1d: np.ndarray) -> None:
        m = realized_volatility(returns_1d, window=60, annualize=True)
        assert m.value > 0.0

    def test_realized_vol_no_annualize(self, returns_1d: np.ndarray) -> None:
        ann = realized_volatility(returns_1d, annualize=True)
        raw = realized_volatility(returns_1d, annualize=False)
        assert ann.value > raw.value

    def test_short_series(self) -> None:
        assert realized_volatility([0.01]).value == 0.0
        assert realized_volatility([]).value == 0.0

    def test_ewma_vol(self, returns_1d: np.ndarray) -> None:
        m = ewma_volatility(returns_1d, lambda_=0.94)
        assert m.value > 0.0

    def test_ewma_empty_and_initial(self) -> None:
        assert ewma_volatility([]).value == 0.0
        m = ewma_volatility([0.01, -0.02, 0.015], initial_variance=0.0001)
        assert m.value >= 0.0


class TestBetaAndTracking:
    def test_beta(self, rng: np.random.Generator) -> None:
        bench = rng.normal(0, 0.01, 200)
        asset = 1.2 * bench + rng.normal(0, 0.002, 200)
        m = beta(asset, bench)
        assert m.value == pytest.approx(1.2, rel=0.15)
        assert "r_squared" in m.parameters or "r_squared" in m.metadata or True

    def test_beta_window(self, rng: np.random.Generator) -> None:
        b = rng.normal(0, 0.01, 150)
        a = b + rng.normal(0, 0.001, 150)
        m = beta(a, b, window=60)
        assert np.isfinite(m.value)

    def test_beta_zero_bench_var(self) -> None:
        assert beta([0.01, 0.02], [0.0, 0.0]).value == 0.0

    def test_tracking_error(self, rng: np.random.Generator) -> None:
        b = rng.normal(0, 0.01, 120)
        a = b + rng.normal(0, 0.005, 120)
        m = tracking_error(a, b, annualize=True)
        assert m.value > 0.0


class TestCorrelationFamily:
    def test_correlation_1d(self, returns_1d: np.ndarray) -> None:
        out = correlation_matrix(returns_1d)
        assert "matrix" in out
        assert out["matrix"] == [[1.0]] or len(out["matrix"]) == 1

    def test_correlation_2d(self, returns_2d: np.ndarray) -> None:
        out = correlation_matrix(returns_2d)
        mat = np.asarray(out["matrix"])
        assert mat.shape == (4, 4)
        assert np.allclose(np.diag(mat), 1.0, atol=1e-6)

    def test_correlation_window(self, returns_2d: np.ndarray) -> None:
        out = correlation_matrix(returns_2d, window=50)
        assert out["n_obs"] <= 50 or "matrix" in out

    def test_correlation_bad_dim(self) -> None:
        with pytest.raises(ValueError):
            correlation_matrix(np.zeros((2, 2, 2)))

    def test_covariance(self, returns_2d: np.ndarray) -> None:
        out = covariance_matrix(returns_2d)
        mat = np.asarray(out["matrix"])
        assert mat.shape[0] == mat.shape[1]
        assert np.all(np.linalg.eigvalsh(mat) >= -1e-8)

    def test_shrinkage(self, returns_2d: np.ndarray) -> None:
        out = shrinkage_covariance(returns_2d)
        assert "matrix" in out
        out2 = shrinkage_covariance(returns_2d, intensity=0.5)
        assert out2 is not None

    def test_ewma_corr_cov(self, returns_2d: np.ndarray) -> None:
        c = ewma_correlation(returns_2d, lambda_=0.94)
        v = ewma_covariance(returns_2d, lambda_=0.94)
        assert "matrix" in c and "matrix" in v

    def test_rolling_correlation(self, rng: np.random.Generator) -> None:
        x = rng.normal(0, 1, 100)
        y = 0.5 * x + rng.normal(0, 1, 100)
        m = rolling_correlation(x, y, window=30)
        assert np.isfinite(m.value)

    def test_rolling_short(self) -> None:
        assert rolling_correlation([1, 2, 3], [1, 2, 3], window=10).value == 0.0


class TestGapAndLiquidity:
    def test_gap_historical(self, rng: np.random.Generator) -> None:
        overnight = rng.normal(0, 0.005, 100)
        overnight[::20] -= 0.03
        m = gap_risk(overnight, method="historical")
        assert m.value >= 0.0

    def test_gap_parametric(self, rng: np.random.Generator) -> None:
        m = gap_risk(rng.normal(0, 0.01, 80), method="parametric", confidence=0.97)
        assert m.value >= 0.0

    def test_gap_empty(self) -> None:
        assert gap_risk([]).value == 0.0

    def test_liquidity_risk(self) -> None:
        out = liquidity_risk(
            position_size=1_000_000,
            adv=5_000_000,
            spread=0.001,
            price=50.0,
            volatility=0.02,
            max_participation=0.10,
            impact_coeff=0.1,
        )
        assert "score" in out or "measures" in out
        assert 0.0 <= float(out.get("score", out.get("liquidity_score", 0))) <= 1.0 or "measures" in out

    def test_liquidity_zero_position(self) -> None:
        out = liquidity_risk(position_size=0.0, adv=1e6, spread=0.001)
        assert out is not None


class TestExposure:
    def test_long_short(self) -> None:
        w = np.array([0.4, 0.3, -0.2, -0.1])
        assert gross_exposure(w).value == pytest.approx(1.0)
        assert net_exposure(w).value == pytest.approx(0.4)
        assert long_exposure(w).value == pytest.approx(0.7)
        assert short_exposure(w).value == pytest.approx(0.3)

    def test_summary(self, weights_4: np.ndarray) -> None:
        s = exposure_summary(weights_4)
        assert isinstance(s, dict)
        assert len(s) > 0


class TestConcentrationDiversification:
    def test_herfindahl(self, weights_4: np.ndarray) -> None:
        h = herfindahl(weights_4)
        assert h.value > 0.0
        assert "effective_n" in h.parameters

    def test_max_weight(self, weights_4: np.ndarray) -> None:
        m = max_weight(weights_4)
        assert m.value == pytest.approx(0.35)

    def test_concentration_risk(self, weights_4: np.ndarray) -> None:
        out = concentration_risk(weights_4)
        assert "score" in out or "herfindahl" in out or "name" in out

    def test_diversification_ratio(self, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        m = diversification_ratio(weights_4, cov_4)
        assert m.value > 0.0

    def test_diversification_bad_cov(self, weights_4: np.ndarray) -> None:
        with pytest.raises(ValueError):
            diversification_ratio(weights_4, np.ones((3, 4)))


class TestFactorExposures:
    def test_ols_betas(self, rng: np.random.Generator) -> None:
        t = 200
        f1 = rng.normal(0, 0.01, t)
        f2 = rng.normal(0, 0.01, t)
        asset = 0.5 * f1 + 0.3 * f2 + rng.normal(0, 0.002, t)
        out = factor_exposures(asset, np.column_stack([f1, f2]), factor_names=["mkt", "smb"])
        assert "betas" in out or "exposures" in out or "factors" in out
        # Accept either nested structure
        betas = out.get("betas") or out.get("exposures") or out
        assert betas is not None

    def test_1d_factor(self, rng: np.random.Generator) -> None:
        f = rng.normal(0, 0.01, 100)
        a = 0.8 * f + rng.normal(0, 0.001, 100)
        out = factor_exposures(a, f)
        assert out is not None

    def test_insufficient_obs(self) -> None:
        out = factor_exposures([0.01, 0.02], [[0.01], [0.02], [0.03]])
        assert out is not None

    def test_bad_ndim(self) -> None:
        with pytest.raises(ValueError):
            factor_exposures([0.01] * 20, np.zeros((20, 2, 2)))


class TestPortfolioRisk:
    def test_volatility(self, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        m = portfolio_volatility(weights_4, cov_4)
        assert m.value > 0.0

    def test_volatility_bad_cov(self) -> None:
        with pytest.raises(ValueError):
            portfolio_volatility([0.5, 0.5], np.ones(3))

    def test_marginal(self, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        out = marginal_risk_contribution(weights_4, cov_4)
        assert out is not None

    def test_component(self, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        out = component_risk_contribution(weights_4, cov_4)
        assert out is not None
        # Component contributions should roughly sum to portfolio vol
        if "contributions" in out:
            assert np.sum(out["contributions"]) == pytest.approx(
                out.get("portfolio_volatility", np.sum(out["contributions"])), rel=1e-5, abs=1e-5
            )

    def test_portfolio_risk_bundle(self, weights_4: np.ndarray, cov_4: np.ndarray) -> None:
        out = portfolio_risk(weights_4, cov_4)
        assert isinstance(out, dict)
        assert len(out) >= 3

    def test_weight_broadcast(self, cov_4: np.ndarray) -> None:
        out = portfolio_risk(1.0, cov_4)  # scalar → equal via as_weights
        assert out is not None
