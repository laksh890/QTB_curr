"""Shared fixtures for portfolio unit tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from iqrp.app.portfolio import PortfolioConstructionEngine, PortfolioSettings


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def n_assets() -> int:
    return 4


@pytest.fixture
def names(n_assets: int) -> list[str]:
    return [f"A{i}" for i in range(n_assets)]


@pytest.fixture
def returns(rng: np.random.Generator, n_assets: int) -> np.ndarray:
    """Synthetic asset returns; fixed seed, T=200, N<=6."""
    corr = 0.25 * np.ones((n_assets, n_assets)) + 0.75 * np.eye(n_assets)
    vols = np.array([0.012, 0.015, 0.010, 0.018][:n_assets])
    cov = np.outer(vols, vols) * corr
    mu = np.array([0.0004, 0.0006, 0.0003, 0.0005][:n_assets])
    return rng.multivariate_normal(mu, cov, size=200)


@pytest.fixture
def returns_6(rng: np.random.Generator) -> np.ndarray:
    n = 6
    corr = 0.2 * np.ones((n, n)) + 0.8 * np.eye(n)
    vols = rng.uniform(0.008, 0.02, size=n)
    cov = np.outer(vols, vols) * corr
    return rng.multivariate_normal(np.zeros(n), cov, size=250)


@pytest.fixture
def mu(returns: np.ndarray) -> np.ndarray:
    return np.mean(returns, axis=0)


@pytest.fixture
def cov(returns: np.ndarray) -> np.ndarray:
    return np.cov(returns, rowvar=False)


@pytest.fixture
def weights(n_assets: int) -> np.ndarray:
    w = np.ones(n_assets) / n_assets
    return w


@pytest.fixture
def current_weights(n_assets: int) -> np.ndarray:
    w = np.array([0.4, 0.3, 0.2, 0.1][:n_assets], dtype=float)
    w = w / w.sum()
    return w


@pytest.fixture
def portfolio_settings() -> PortfolioSettings:
    """Fast defaults: risk validation off."""
    return PortfolioSettings(
        require_risk_validation=False,
        seed=42,
        method="mean_variance",
        long_only=True,
        max_weight=0.5,
        max_gross=1.5,
        max_leverage=2.0,
        max_turnover=0.5,
        risk_aversion=1.0,
        fallback="current",
    )


@pytest.fixture
def engine(portfolio_settings: PortfolioSettings) -> PortfolioConstructionEngine:
    return PortfolioConstructionEngine(settings=portfolio_settings)


@pytest.fixture
def risk_settings_on() -> PortfolioSettings:
    return PortfolioSettings(
        require_risk_validation=True,
        seed=42,
        method="min_variance",
        long_only=True,
        max_weight=0.5,
        fallback="cash",
    )


@pytest.fixture
def rejecting_risk_engine() -> Any:
    """Minimal stub: Risk rejects oversized max-abs weight."""

    class _Breach:
        def __init__(self, severity: str = "hard") -> None:
            self.severity = type("S", (), {"value": severity})()

        def to_dict(self) -> dict[str, Any]:
            return {"severity": "hard", "name": "max_position"}

    class _Decision:
        def __init__(self, approved: bool, action: str, reason: str) -> None:
            self.approved = approved
            self.action = action
            self.reason = reason

        def to_dict(self) -> dict[str, Any]:
            return {
                "approved": self.approved,
                "action": self.action,
                "reason": self.reason,
                "status": self.action,
            }

    class RejectingRisk:
        def check_limits(self, weights: Any = None, **kwargs: Any) -> list[Any]:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.size and float(np.max(np.abs(w))) > 0.35:
                return [_Breach("hard")]
            return []

        def validate_position(
            self,
            *,
            proposed_weight: float,
            weights: Any,
            returns: Any = None,
            forecast_confidence: float = 0.0,
            asset_index: int = 0,
            **kwargs: Any,
        ) -> _Decision:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if float(np.max(np.abs(w))) > 0.35 or abs(proposed_weight) > 0.35:
                return _Decision(False, "REJECT", "oversized position")
            return _Decision(True, "APPROVE", "ok")

    return RejectingRisk()


@pytest.fixture
def approving_risk_engine() -> Any:
    class _Decision:
        def to_dict(self) -> dict[str, Any]:
            return {"approved": True, "action": "APPROVE", "reason": "ok", "status": "APPROVE"}

    class ApprovingRisk:
        def check_limits(self, weights: Any = None, **kwargs: Any) -> list[Any]:
            return []

        def validate_position(self, **kwargs: Any) -> _Decision:
            return _Decision()

    return ApprovingRisk()


@pytest.fixture
def forecasts(n_assets: int) -> np.ndarray:
    return np.array([0.01, 0.02, -0.005, 0.015][:n_assets], dtype=float)


@pytest.fixture
def signals(n_assets: int) -> np.ndarray:
    return np.array([1.2, -0.5, 0.8, 0.3][:n_assets], dtype=float)


@pytest.fixture
def prices(n_assets: int) -> np.ndarray:
    return np.array([100.0, 50.0, 25.0, 80.0][:n_assets], dtype=float)


@pytest.fixture
def adv(n_assets: int) -> np.ndarray:
    return np.array([1e7, 5e6, 2e6, 8e6][:n_assets], dtype=float)
