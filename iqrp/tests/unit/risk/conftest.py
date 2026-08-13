"""Shared fixtures for risk unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings
from iqrp.app.risk.capital import CapitalAllocator, CapitalSettings
from iqrp.app.risk.config import (
    LeverageConfig,
    LimitConfig,
    MonteCarloConfig,
    SizingConfig,
    VaRConfig,
)
from iqrp.app.risk.ensemble import EnsembleSettings, RiskIntelligenceEnsemble
from iqrp.app.risk.ensemble.config import HysteresisConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def returns_1d(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0005, 0.01, size=300)


@pytest.fixture
def returns_2d(rng: np.random.Generator) -> np.ndarray:
    corr = 0.3 * np.ones((4, 4)) + 0.7 * np.eye(4)
    cov = corr * (0.015**2)
    return rng.multivariate_normal(np.zeros(4), cov, size=250)


@pytest.fixture
def weights_4() -> np.ndarray:
    return np.array([0.35, 0.25, 0.20, 0.20])


@pytest.fixture
def cov_4(returns_2d: np.ndarray) -> np.ndarray:
    return np.cov(returns_2d, rowvar=False)


@pytest.fixture
def fast_settings() -> RiskSettings:
    """Settings with small Monte Carlo budgets for fast unit tests."""
    return RiskSettings(
        seed=42,
        var=VaRConfig(method="historical", confidence=0.95, n_simulations=200),
        monte_carlo=MonteCarloConfig(n_simulations=200, seed=42, block_size=5),
        sizing=SizingConfig(max_kelly=0.5, max_leverage=2.0, target_volatility=0.10),
        limits=LimitConfig(max_position=0.10, max_leverage=2.0),
        leverage=LeverageConfig(max_leverage=2.0, min_leverage=0.0, confidence_cap=1.25),
    )


@pytest.fixture
def engine(fast_settings: RiskSettings) -> RiskIntelligenceEngine:
    return RiskIntelligenceEngine(settings=fast_settings)


# ---- Phase 09 capital / ensemble fixtures ---------------------------------


@pytest.fixture
def strategy_names() -> list[str]:
    return ["alpha", "beta", "gamma", "delta"]


@pytest.fixture
def capital_returns(rng: np.random.Generator) -> np.ndarray:
    """4-strategy returns with moderate correlation; fixed seed."""
    corr = 0.25 * np.ones((4, 4)) + 0.75 * np.eye(4)
    cov = corr * (0.012**2)
    return rng.multivariate_normal(np.zeros(4), cov, size=200)


@pytest.fixture
def capital_cov(capital_returns: np.ndarray) -> np.ndarray:
    return np.cov(capital_returns, rowvar=False)


@pytest.fixture
def crowded_returns(rng: np.random.Generator) -> np.ndarray:
    """Highly correlated strategies for crowding tests."""
    corr = 0.85 * np.ones((4, 4)) + 0.15 * np.eye(4)
    cov = corr * (0.015**2)
    return rng.multivariate_normal(np.zeros(4), cov, size=180)


@pytest.fixture
def capital_settings() -> CapitalSettings:
    return CapitalSettings(
        seed=42,
        max_weight=0.40,
        max_concentration=0.40,
        max_leverage=2.0,
        max_gross_exposure=1.5,
        max_participation=0.10,
        correlation_crowding_threshold=0.60,
        correlation_scale_floor=0.25,
        missing_capacity_scale=0.50,
        missing_liquidity_scale=0.50,
    )


@pytest.fixture
def capital_allocator(capital_settings: CapitalSettings) -> CapitalAllocator:
    return CapitalAllocator(settings=capital_settings)


@pytest.fixture
def ensemble_settings() -> EnsembleSettings:
    return EnsembleSettings(
        seed=42,
        hard_halt_on_single=False,
        min_dimensions_for_halt=2,
        missing_metrics_fallback_state="CAPITAL_PRESERVATION",
        missing_metrics_fallback_action="REJECT",
        hysteresis=HysteresisConfig(
            escalation_confirmations=1,
            recovery_confirmations=3,
            dimension_confirmation_threshold=0.75,
        ),
    )


@pytest.fixture
def ensemble(ensemble_settings: EnsembleSettings) -> RiskIntelligenceEnsemble:
    return RiskIntelligenceEnsemble(settings=ensemble_settings)


@pytest.fixture
def healthy_metrics() -> dict:
    """Complete critical metrics at low risk — should allow APPROVE paths."""
    return {
        "volatility": 0.08,
        "var": 0.02,
        "cvar": 0.03,
        "drawdown": 0.01,
        "liquidity_score": 0.9,
        "concentration": 0.15,
        "correlation": 0.25,
        "model_risk": 0.10,
        "operational": 0.05,
    }


@pytest.fixture
def stressed_metrics() -> dict:
    """Elevated but not single-hot-halt metrics."""
    return {
        "volatility": 0.35,
        "var": 0.08,
        "cvar": 0.12,
        "drawdown": 0.12,
        "liquidity_score": 0.4,
        "concentration": 0.35,
        "correlation": 0.70,
        "model_risk": 0.40,
        "operational": 0.30,
    }
